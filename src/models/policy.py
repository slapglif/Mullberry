import torch
import torch.nn as nn
from typing import List, Dict, Optional
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
    ) -> List[str]:
        """Generate next reasoning steps"""
        raise NotImplementedError
        
    def evaluate(
        self,
        question: str,
        reasoning_path: List[str],
        images: Optional[torch.Tensor] = None
    ) -> float:
        """Evaluate the quality of a reasoning path"""
        raise NotImplementedError

class ReflectivePolicy(PolicyModel):
    """Policy model with reflection capabilities"""
    def __init__(
        self,
        base_model: nn.Module,
        reflection_head: nn.Module
    ):
        super().__init__()
        self.base_model = base_model
        self.reflection_head = reflection_head
        
    def forward(
        self,
        question: str,
        current_path: List[str],
        images: Optional[torch.Tensor] = None
    ) -> List[str]:
        # Get base model prediction
        base_output = self.base_model(
            question,
            current_path,
            images
        )
        
        # Add reflection if needed
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
        # Evaluate base reasoning
        base_score = self.base_model.evaluate(
            question,
            reasoning_path,
            images
        )
        
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
            return (base_score + reflection_score) / 2
            
        return base_score
        
    def should_reflect(self, current_output: List[str]) -> bool:
        """Determine if reflection is needed"""
        reflection_score = self.reflection_head(current_output)
        return reflection_score < 0.5
        
    def generate_reflection(
        self,
        question: str,
        current_path: List[str],
        current_output: List[str]
    ) -> List[str]:
        """Generate reflection steps"""
        reflection_prompt = (
            "The previous reasoning step was incorrect. "
            "Let's correct it with the following steps:"
        )
        return [reflection_prompt]
        
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
                # Check if reflection led to improvement
                before = reasoning_path[marker]
                after = reasoning_path[marker + 2]
                
                reflection_score = self.reflection_head(
                    [before, after]
                )
                reflection_scores.append(reflection_score)
                
        return sum(reflection_scores) / len(reflection_scores)
