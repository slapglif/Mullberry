import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from typing import Optional

def train_mulberry(
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule,
    max_epochs: int = 10,
    checkpoint_dir: Optional[str] = None,
    tensorboard_dir: Optional[str] = None
):
    # Setup callbacks
    callbacks = []

    if checkpoint_dir:
        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="mulberry-{epoch:02d}-{val_accuracy:.2f}",
            monitor="val_accuracy",
            mode="max",
            save_top_k=3
        )
        callbacks.append(checkpoint_callback)

    # Setup logger
    logger = None
    if tensorboard_dir:
        logger = TensorBoardLogger(tensorboard_dir)

    # Initialize trainer without gradient clipping since we use manual optimization
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=callbacks,
        logger=logger,
        accelerator="cpu",
        precision="32-true",
        accumulate_grad_batches=4  # For memory efficiency
    )

    # Train model
    trainer.fit(model, datamodule)