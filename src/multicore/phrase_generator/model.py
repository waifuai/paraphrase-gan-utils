from trax import layers as tl

def create_transformer_model(input_vocab_size=256,
                             output_vocab_size=256,
                             d_model=128,
                             d_ff=512,
                             n_heads=4,
                             n_encoder_layers=2,
                             n_decoder_layers=2,
                             mode="train"):
    """Creates a Transformer model using Trax."""
    
    # Build encoder: embedding + repeated feed-forward blocks + pooling.
    encoder_layers = [tl.Embedding(input_vocab_size, d_model)]
    for _ in range(n_encoder_layers):
        encoder_layers.extend([
            tl.Relu(),
            tl.Dense(d_model),
            tl.LayerNorm()
        ])
    encoder_layers.append(tl.Mean(axis=1))
    encoder = tl.Serial(*encoder_layers)

    # Build decoder: embedding + repeated feed-forward blocks + final dense + log softmax.
    decoder_layers = [tl.Embedding(output_vocab_size, d_model)]
    for _ in range(n_decoder_layers):
        decoder_layers.extend([
            tl.Relu(),
            tl.Dense(d_model),
            tl.LayerNorm()
        ])
    decoder_layers.extend([
        tl.Dense(output_vocab_size),
        tl.LogSoftmax()
    ])
    decoder = tl.Serial(*decoder_layers)

    # The overall model combines encoder and decoder in parallel, adds their outputs,
    # and applies a final log softmax.
    model = tl.Serial(
        tl.Select([0, 0]),
        tl.Parallel(encoder, decoder),
        tl.Add(),
        tl.LogSoftmax()
    )
    return model
