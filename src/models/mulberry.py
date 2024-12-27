import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Dict, List, Any
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
        self.automatic_optimization = False  # Manual optimization for better control

    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Optimize input batch processing
        if "images" in x:
            # Efficient batch processing with einops
            images = rearrange(x["images"], 'b c h w -> (b h) w c')
            x["images"] = images

        return self.base_model(x)

    def training_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int
    ) -> torch.Tensor:
        # Manual optimization
        opt = self.optimizers()
        opt.zero_grad()

        # Unpack batch efficiently
        questions = batch["questions"]
        images = batch.get("images", None)
        labels = batch["labels"]

        # Get model predictions with CPU optimization
        inputs = {
            "questions": questions,
            "images": images
        }
        outputs = self(inputs)

        # CoMCTS search for reasoning paths
        reasoning_paths = []
        reflection_paths = []

        # Parallelize search across CPU cores
        def process_question(q_idx):
            question = questions[q_idx]
            # Initial node
            init_node = {
                "reasoning_path": [],
                "value": 0.0,
                "visits": 0,
                "children": []
            }

            # Search for reasoning path
            best_path, tree = self.comcts.search(question, init_node)

            # Get reflective path
            reflection_path = self.comcts.get_reflective_path(tree, best_path)
            return best_path, reflection_path

        # Process questions in parallel
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=self.cpu_workers) as executor:
            results = list(executor.map(
                process_question,
                range(len(questions))
            ))

        for best_path, reflection_path in results:
            reasoning_paths.append(best_path)
            reflection_paths.append(reflection_path)

        # Calculate losses efficiently using einops
        reasoning_logits = rearrange(
            outputs["logits"],
            'b s v -> (b s) v'
        )
        reasoning_targets = rearrange(
            torch.tensor(reasoning_paths),
            'b s -> (b s)'
        )

        reasoning_loss = self.reasoning_loss(
            reasoning_logits,
            reasoning_targets
        )

        reflection_logits = rearrange(
            outputs["logits"],
            'b s v -> (b s) v'
        )
        reflection_targets = rearrange(
            torch.tensor(reflection_paths),
            'b s -> (b s)'
        )

        reflection_loss = self.reflection_loss(
            reflection_logits,
            reflection_targets
        )

        total_loss = reasoning_loss + reflection_loss

        # Manual optimization
        self.manual_backward(total_loss)
        opt.step()

        # Log metrics
        self.log("train_loss", total_loss)
        self.log("reasoning_loss", reasoning_loss)
        self.log("reflection_loss", reflection_loss)

        return total_loss

    def validation_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int
    ) -> None:
        questions = batch["questions"]
        images = batch.get("images", None)
        labels = batch["labels"]

        inputs = {
            "questions": questions,
            "images": images
        }
        outputs = self(inputs)

        reasoning_paths = []

        # Parallel validation
        def process_validation(q_idx):
            question = questions[q_idx]
            init_node = {
                "reasoning_path": [],
                "value": 0.0,
                "visits": 0,
                "children": []
            }

            best_path, _ = self.comcts.search(question, init_node)
            return best_path

        # Process validation in parallel
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=self.cpu_workers) as executor:
            reasoning_paths = list(executor.map(
                process_validation,
                range(len(questions))
            ))

        # Calculate accuracy efficiently
        accuracy = (
            torch.tensor(reasoning_paths) == labels
        ).float().mean()

        self.log("val_accuracy", accuracy)

    def configure_optimizers(self):
        # Use CPU-optimized AdamW
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            foreach=True  # Enable CPU optimization
        )
        return optimizer