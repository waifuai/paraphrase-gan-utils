import itertools
import random
import numpy as np

def load_parabank_data(filepath="./parabank.tsv"):
    """Loads and preprocesses the ParaBank data."""
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            phrases = line.strip().split("\t")
            if len(phrases) >= 2:
                data.extend(itertools.permutations(phrases, 2))
    return data

def data_generator(data, batch_size=32, loop=True):
    """Generator function that yields batches of phrase pairs."""
    while True:
        batch = [random.choice(data) for _ in range(batch_size)]
        inputs = [pair[0] for pair in batch]
        targets = [pair[1] for pair in batch]
        # In a full implementation, you might perform encoding here.
        yield (inputs, targets, np.ones_like(targets), np.ones_like(targets))
        if not loop:
            break
