# src/models/phrase_transformer.py
from trax import layers as tl
import config  # Import centralized configuration


def create_phrase_model(
    input_vocab_size: int,  # Must be provided, often determined dynamically
    output_vocab_size: int,  # Must be provided, often determined dynamically
    d_model: int = config.PHRASE_GENERATOR_D_MODEL,
    d_ff: int = config.PHRASE_GENERATOR_D_FF,
    n_heads: int = config.PHRASE_GENERATOR_N_HEADS,
    n_encoder_layers: int = config.PHRASE_GENERATOR_N_ENCODER_LAYERS,
    n_decoder_layers: int = config.PHRASE_GENERATOR_N_DECODER_LAYERS,
    mode: str = "train",  # Typically 'train', 'eval', or 'predict'
) -> tl.Serial:
    """
    Creates a Transformer-like model using manual Trax layers,
    configured for the Phrase Generator task, using defaults from config.py.

    Note: This model differs significantly from the standard trax.models.Transformer.
          It uses simpler feed-forward blocks instead of full attention blocks.

    Args:
        input_vocab_size: The size of the input vocabulary.
        output_vocab_size: The size of the output vocabulary.
        d_model: Dimensionality of embeddings and hidden layers.
        d_ff: Dimensionality of the feed-forward layer (though not used in original Relu->Dense block).
        n_heads: Number of attention heads (parameter exists but not used in original blocks).
        n_encoder_layers: Number of encoder blocks.
        n_decoder_layers: Number of decoder blocks.
        mode: 'train', 'eval', or 'predict'. Currently unused by the layer definitions.

    Returns:
        An instance of tl.Serial representing the model.
    """
    print(
        f"Creating Phrase model (mode: {mode}, input_vocab: {input_vocab_size}, output_vocab: {output_vocab_size})"
    )

    # Build encoder: embedding + repeated feed-forward blocks + pooling.
    # Original model used Relu -> Dense -> LayerNorm. d_ff and n_heads were unused.
    encoder_layers = [tl.Embedding(input_vocab_size, d_model)]
    for _ in range(n_encoder_layers):
        encoder_layers.extend(
            [
                tl.Relu(),
                tl.Dense(d_model),  # Original didn't use d_ff here
                tl.LayerNorm(),
            ]
        )
    encoder_layers.append(tl.Mean(axis=1))  # Average pooling over sequence length
    encoder = tl.Serial(*encoder_layers)

    # Build decoder: embedding + repeated feed-forward blocks + final dense + log softmax.
    decoder_layers = [tl.Embedding(output_vocab_size, d_model)]
    for _ in range(n_decoder_layers):
        decoder_layers.extend(
            [
                tl.Relu(),
                tl.Dense(d_model),  # Original didn't use d_ff here
                tl.LayerNorm(),
            ]
        )
    decoder_layers.extend([tl.Dense(output_vocab_size), tl.LogSoftmax()])
    decoder = tl.Serial(*decoder_layers)

    # The overall model combines encoder and decoder in parallel, adds their outputs,
    # and applies a final log softmax. This structure seems unusual for seq2seq.
    # Replicating original structure:
    model = tl.Serial(
        tl.Select([0, 0]),  # Duplicate input for parallel branches
        tl.Parallel(encoder, decoder),  # Apply encoder and decoder to the *same* input?
        tl.Add(),  # Add the outputs of encoder and decoder
        tl.LogSoftmax(),  # Final activation
    )
    # Consider if this architecture is intended. A standard Transformer would feed
    # encoder output to the decoder's cross-attention. This adds pooled encoder
    # output to the decoder output before LogSoftmax.

    return model
