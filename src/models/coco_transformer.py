# src/models/coco_transformer.py
from trax import models
from trax import (
    layers as tl,
)  # Keep tl import if needed by models.Transformer/Encoder implicitly or for future use
import config  # Import centralized configuration


def create_coco_model(
    mode: str,
    vocab_size: int,
    model_name: str = config.COCO_MODEL_NAME,
    d_model: int = config.COCO_D_MODEL,
    d_ff: int = config.COCO_D_FF,
    n_heads: int = config.COCO_N_HEADS,
    n_encoder_layers: int = config.COCO_N_ENCODER_LAYERS,
    n_decoder_layers: int = config.COCO_N_DECODER_LAYERS,
    dropout: float = config.COCO_DROPOUT,
) -> models.Transformer | models.TransformerEncoder:
    """
    Creates and returns a Trax Transformer or TransformerEncoder model
    configured for the MS COCO dataset, using defaults from config.py.

    Args:
        mode: 'train', 'eval', or 'predict'.
        vocab_size: The size of the input vocabulary.
        model_name: Name of the model ('transformer' or 'transformer_encoder').
        d_model: Dimensionality of embeddings and hidden layers.
        d_ff: Dimensionality of the feed-forward layer.
        n_heads: Number of attention heads.
        n_encoder_layers: Number of encoder layers.
        n_decoder_layers: Number of decoder layers (used only if model_name='transformer').
        dropout: Dropout rate.

    Returns:
        An instance of trax.models.Transformer or trax.models.TransformerEncoder.

    Raises:
        ValueError: If model_name is unknown.
    """
    print(f"Creating COCO model: {model_name} (mode: {mode}, vocab_size: {vocab_size})")
    if model_name == "transformer":
        model = models.Transformer(
            input_vocab_size=vocab_size,
            d_model=d_model,
            d_ff=d_ff,
            n_heads=n_heads,
            n_encoder_layers=n_encoder_layers,
            n_decoder_layers=n_decoder_layers,
            dropout=dropout,
            mode=mode,
        )
    elif model_name == "transformer_encoder":
        # Note: TransformerEncoder doesn't have n_decoder_layers parameter
        model = models.TransformerEncoder(
            input_vocab_size=vocab_size,
            d_model=d_model,
            d_ff=d_ff,
            n_heads=n_heads,
            n_layers=n_encoder_layers,  # Uses n_layers instead of n_encoder_layers
            dropout=dropout,
            mode=mode,
        )
    else:
        raise ValueError(
            f"Unknown COCO model name: {model_name}. Choose 'transformer' or 'transformer_encoder'."
        )
    return model
