import os
from trax import lr as lr_lib
from trax import layers as tl
from trax.supervised import training
import trax.optimizers

def create_tasks(train_gen, eval_gen, learning_rate):
    """Creates training and evaluation tasks."""
    lr_schedule = lr_lib.warmup_and_rsqrt_decay(n_warmup_steps=1000, max_value=learning_rate)
    
    train_task = training.TrainTask(
        labeled_data=train_gen,
        loss_layer=tl.WeightedCategoryCrossEntropy(),
        optimizer=trax.optimizers.Adam(learning_rate),
        lr_schedule=lr_schedule,
        n_steps_per_checkpoint=500,
    )

    eval_task = training.EvalTask(
        labeled_data=eval_gen,
        metrics=[tl.WeightedCategoryCrossEntropy(), tl.WeightedCategoryAccuracy()],
    )

    return train_task, eval_task

def train_model(model, train_task, eval_task, output_dir, n_steps=10000):
    """Trains the Transformer model."""
    training_loop = training.Loop(
        model,
        train_task,
        eval_tasks=[eval_task],
        output_dir=output_dir,
    )
    training_loop.run(n_steps=n_steps)
    return training_loop
