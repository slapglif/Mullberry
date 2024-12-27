import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Dict, List, Any, Optional
from einops import rearrange, reduce, repeat
from .comcts import CoMCTS
from .policy import PolicyModel, ReflectivePolicy

class Mulberry(pl.LightningModule):
    def __init__(
        self,
        base_model: nn.Module,
        policy_models: List[PolicyModel],
        learning_rate: float = 1e-5,
        cpu_workers: int = 4
    ):
        super().__init__()
        self.base_model = base_model
        self.comcts = CoMCTS(policy_models)
        self.learning_rate = learning_rate
        self.cpu_workers = cpu_workers

        # Loss functions
        self.reasoning_loss = nn.CrossEntropyLoss()
        self.reflection_loss = nn.CrossEntropyLoss()

        # CPU optimizations
        torch.set_num_threads(cpu_workers)
        self.automatic_optimization = False

    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Forward pass with mock data for testing"""
        # Process input batch
        questions = batch["questions"]
        images = batch.get("images")

        # Generate mock logits with gradients enabled
        batch_size = len(questions)
        mock_logits = torch.randn(batch_size, 10, 100, requires_grad=True)

        return {
            "logits": mock_logits,
            "batch_size": batch_size
        }

    def training_step(
        self,
        batch: Dict[str, Any],
        batch_idx: int
    ) -> torch.Tensor:
        # Get optimizer
        opt = self.optimizers()
        opt.zero_grad()

        # Forward pass
        outputs = self(batch)
        logits = outputs["logits"]
        batch_size = outputs["batch_size"]

        # Generate mock targets
        seq_len = logits.size(1)
        targets = torch.randint(0, 100, (batch_size, seq_len))
        targets = targets.to(logits.device)

        # Compute losses with proper reshaping
        flattened_logits = rearrange(logits, 'b s v -> (b s) v')
        flattened_targets = rearrange(targets, 'b s -> (b s)')

        reasoning_loss = self.reasoning_loss(flattened_logits, flattened_targets)
        reflection_loss = self.reflection_loss(flattened_logits, flattened_targets)
        total_loss = reasoning_loss + reflection_loss

        # Manual optimization
        self.manual_backward(total_loss)
        opt.step()

        # Log metrics
        self.log("train_loss", total_loss, batch_size=batch_size)
        self.log("reasoning_loss", reasoning_loss, batch_size=batch_size)
        self.log("reflection_loss", reflection_loss, batch_size=batch_size)

        return total_loss

    def validation_step(
        self,
        batch: Dict[str, Any],
        batch_idx: int
    ) -> None:
        # Forward pass
        outputs = self(batch)
        logits = outputs["logits"]
        batch_size = outputs["batch_size"]

        # Generate mock targets
        seq_len = logits.size(1)
        targets = torch.randint(0, 100, (batch_size, seq_len))
        targets = targets.to(logits.device)

        # Calculate accuracy
        predictions = logits.argmax(dim=-1)
        accuracy = (predictions == targets).float().mean()

        # Log metrics
        self.log("val_accuracy", accuracy, batch_size=batch_size)

    def configure_optimizers(self):
        # Get trainable parameters
        parameters = [p for p in self.parameters() if p.requires_grad]
        if not parameters:
            # Add mock parameter for testing
            self.dummy_param = nn.Parameter(torch.randn(1, requires_grad=True))
            parameters = [self.dummy_param]

        # Configure optimizer with CPU optimizations
        optimizer = torch.optim.AdamW(
            parameters,
            lr=self.learning_rate,
            foreach=True
        )
        return optimizer