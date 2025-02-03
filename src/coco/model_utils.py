import os
import functools
import jax
from trax import models, training, optimizers, layers as tl
from trax import data

def create_model(mode, vocab_size, model_name='transformer', d_model=512, d_ff=2048,
                 n_heads=8, n_encoder_layers=6, n_decoder_layers=6, dropout=0.1):
    """
    Creates and returns a Transformer model.
    """
    if model_name == 'transformer':
        model = models.Transformer(
            input_vocab_size=vocab_size,
            d_model=d_model,
            d_ff=d_ff,
            n_heads=n_heads,
            n_encoder_layers=n_encoder_layers,
            n_decoder_layers=n_decoder_layers,
            dropout=dropout,
            mode=mode
        )
    elif model_name == 'transformer_encoder':
        model = models.TransformerEncoder(
            input_vocab_size=vocab_size,
            d_model=d_model,
            d_ff=d_ff,
            n_heads=n_heads,
            n_layers=n_encoder_layers,
            dropout=dropout,
            mode=mode
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    return model

def train_task_setup(model, train_stream, eval_stream, output_dir, train_steps, eval_steps, learning_rate):
    """
    Sets up and returns a training loop.
    """
    lr_schedule = optimizers.lr_schedules.warmup_and_rsqrt_decay(
        n_warmup_steps=1000, max_value=learning_rate
    )

    train_task = training.TrainTask(
        labeled_data=train_stream(),
        loss_layer=tl.CrossEntropyLoss(),
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        lr_schedule=lr_schedule,
        n_steps_per_checkpoint=eval_steps,
    )

    eval_task = training.EvalTask(
        labeled_data=eval_stream(),
        metrics=[tl.CrossEntropyLoss(), tl.Accuracy()],
    )

    checkpoint_at = [train_steps // (eval_steps * 2), train_steps]
    checkpoint_low_at = [train_steps // (eval_steps * 4)]

    return training.Loop(
        model,
        train_task,
        eval_tasks=[eval_task],
        output_dir=output_dir,
        checkpoint_at=checkpoint_at,
        checkpoint_low_at=checkpoint_low_at
    )

def decode(model, input_sentence, vocab, data_dir, vocab_filename, output_dir, max_len=256):
    """
    Decodes (paraphrases) an input sentence using a trained model.
    
    The function loads model weights from a checkpoint file located in output_dir.
    """
    # Recreate the model in eval mode.
    model = create_model('eval', len(vocab) + 1, model_name=model.name)
    model_path = os.path.join(output_dir, "model.pkl.gz")
    model.init_from_file(model_path, weights_only=True)

    # Tokenize the input sentence.
    vocab_file = os.path.join(data_dir, vocab_filename)
    token_gen = data.tokenize(iter([input_sentence]), vocab_file=vocab_file, n_reserved_ids=0)
    # Convert token generator to list (assuming a single sentence)
    input_ids = next(token_gen)
    # Pad to max_len.
    input_ids = input_ids + [0] * (max_len - len(input_ids))
    inputs = jax.numpy.array(input_ids)[None, :]

    # Use Trax's fast decode.
    output_ids = models.transformer.fast_decode(
        model, inputs,
        start_id=ord('\n'),
        eos_id=ord('\n'),
        max_len=max_len,
        temperature=0.0,
        n_beams=1
    )
    output_ids = output_ids[0].tolist()
    # Remove padding tokens.
    output_ids = [idx for idx in output_ids if idx != 0]
    # Detokenize: convert IDs back to characters.
    output_sentence = "".join([chr(c) for c in output_ids if c != ord('\n')])
    return output_sentence
