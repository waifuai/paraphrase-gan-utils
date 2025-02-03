import os
import random
import numpy as np
import trax
from phrase_generator.data import load_parabank_data, data_generator
from phrase_generator.model import create_transformer_model
from phrase_generator.trainer.train import create_tasks, train_model

def set_seed(seed):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    trax.supervised.trainer_lib.init_random_number_generators(seed)

def main():
    # Set seed for reproducibility
    set_seed(42)

    # Load and preprocess data
    parabank_data = load_parabank_data(filepath="./parabank.tsv")
    
    # Split data into training and evaluation sets
    train_size = int(0.9 * len(parabank_data))
    train_data = parabank_data[:train_size]
    eval_data = parabank_data[train_size:]

    # Create data generators
    batch_size = 32
    train_gen = data_generator(train_data, batch_size)
    eval_gen = data_generator(eval_data, batch_size)

    # Define hyperparameters
    hparams = {
        "d_model": 128,
        "d_ff": 512,
        "n_heads": 4,
        "n_encoder_layers": 2,
        "n_decoder_layers": 2,
        "learning_rate": 0.05,
    }

    # Create the Transformer model
    model = create_transformer_model(
        d_model=hparams["d_model"],
        d_ff=hparams["d_ff"],
        n_heads=hparams["n_heads"],
        n_encoder_layers=hparams["n_encoder_layers"],
        n_decoder_layers=hparams["n_decoder_layers"],
    )

    # Create training and evaluation tasks
    train_task, eval_task = create_tasks(train_gen, eval_gen, hparams["learning_rate"])

    # Define and prepare the output directory
    output_dir = os.path.expanduser("~/output_dir/")
    if os.path.exists(output_dir):
        import shutil
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Start training (adjust n_steps as desired)
    n_steps = 1000
    train_model(model, train_task, eval_task, output_dir=output_dir, n_steps=n_steps)

if __name__ == "__main__":
    main()
