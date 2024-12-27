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

    # Initialize trainer with CPU-optimized settings
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=callbacks,
        logger=logger,
        accelerator="cpu",
        devices=1,
        precision="32-true",
        enable_progress_bar=True,
        enable_model_summary=True,
        log_every_n_steps=1,  # More frequent logging for testing
        num_sanity_val_steps=0,  # Skip validation sanity check for testing
        deterministic=True  # Ensure reproducibility
    )

    # Train model
    trainer.fit(model, datamodule)