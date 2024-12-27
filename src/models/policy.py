import torch
import torch.nn as nn
from typing import List, Dict, Optional, Union, Tuple, Any
from einops import rearrange, reduce, repeat

class PolicyModel(nn.Module):
    """Base class for policy models used in collective learning"""
    def __init__(self):
        super().__init__()

    def forward(
        self,
        question: str,
        current_path: List[str],
        images: Optional[torch.Tensor] = None
    ) -> Union[List[str], Dict[str, torch.Tensor]]:
        """Generate next reasoning steps or training logits"""
        raise NotImplementedError

    def evaluate(
        self,
        question: str,
        reasoning_path: List[str],
        images: Optional[torch.Tensor] = None
    ) -> float:
        """Evaluate the quality of a reasoning path"""
        raise NotImplementedError

    def predict_step(
        self,
        question: str,
        current_path: List[str],
        images: Optional[torch.Tensor] = None
    ) -> List[str]:
        """Generate next reasoning steps for inference"""
        output = self.forward(question, current_path, images)
        if isinstance(output, dict):
            # Convert logits to text predictions
            logits = output["logits"]  # Shape: [batch_size, seq_len, vocab_size]
            predictions = logits[0].argmax(dim=-1)  # Shape: [seq_len]
            return [f"Step {i+1}: {p.item()}" for i, p in enumerate(predictions)]
        return output

class ReflectivePolicy(PolicyModel):
    """Policy model with reflection capabilities"""
    def __init__(
        self,
        base_model: nn.Module,
        reflection_head: nn.Module,
        hidden_size: int = 768,
        embedding_size: int = 10
    ):
        super().__init__()
        self.base_model = base_model
        self.reflection_head = reflection_head
        self.text_embedding = nn.Linear(hidden_size, embedding_size)
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size

    def _mock_text_embedding(self, text: Union[str, List[str]]) -> torch.Tensor:
        """Generate mock embeddings for testing"""
        if isinstance(text, str):
            batch_size = 1
        else:
            batch_size = len(text)

        # Generate mock text embeddings
        text_tensor = torch.randn(batch_size, self.hidden_size)
        return self.text_embedding(text_tensor)

    def forward(
        self,
        question: str,
        current_path: List[str],
        images: Optional[torch.Tensor] = None
    ) -> Union[List[str], Dict[str, torch.Tensor]]:
        # Get base model prediction
        base_output = self.base_model(question, current_path, images)

        # For training mode, return logits directly
        if isinstance(base_output, dict):
            return base_output

        # For inference mode, handle reflection
        if self.should_reflect(base_output):
            reflection = self.generate_reflection(
                question,
                current_path,
                base_output
            )
            return reflection + base_output

        return base_output

    def evaluate(
        self,
        question: str,
        reasoning_path: List[str],
        images: Optional[torch.Tensor] = None
    ) -> float:
        # Use predict_step for evaluation to ensure text output
        base_output = self.base_model.predict_step(question, reasoning_path, images)

        # Add reflection score if present
        reflection_markers = [
            i for i, step in enumerate(reasoning_path)
            if "incorrect" in step.lower()
        ]

        if reflection_markers:
            reflection_score = self.evaluate_reflection(
                question,
                reasoning_path,
                reflection_markers
            )
            return (self.base_model.evaluate(question, base_output, images) + reflection_score) / 2

        return self.base_model.evaluate(question, base_output, images)

    def should_reflect(self, current_output: List[str]) -> bool:
        """Determine if reflection is needed"""
        # Generate embeddings for current output
        embeddings = self._mock_text_embedding(current_output)
        reflection_score = self.reflection_head(embeddings).mean()
        return reflection_score.item() < 0.5

    def generate_reflection(
        self,
        question: str,
        current_path: List[str],
        current_output: List[str]
    ) -> List[str]:
        """Generate reflection steps"""
        reflection_prompt = [
            "The previous reasoning step was incorrect.",
            "Let's correct it with the following steps:"
        ]
        return reflection_prompt

    def evaluate_reflection(
        self,
        question: str,
        reasoning_path: List[str],
        reflection_markers: List[int]
    ) -> float:
        """Evaluate quality of reflection steps"""
        reflection_scores = []

        for marker in reflection_markers:
            if marker + 2 < len(reasoning_path):
                # Generate embeddings for reflection step
                embeddings = self._mock_text_embedding(
                    reasoning_path[marker:marker + 3]
                )
                reflection_score = self.reflection_head(embeddings).mean()
                reflection_scores.append(reflection_score.item())

        return sum(reflection_scores) / len(reflection_scores) if reflection_scores else 0.0