import torch
from typing import List, Dict

def compute_search_metrics(trees: List[Dict]) -> Dict[str, float]:
    """Compute metrics for CoMCTS search"""
    total_iterations = 0
    successful_searches = 0
    avg_path_length = 0
    
    for tree in trees:
        # Count iterations
        total_iterations += sum(
            node["visits"] for node in tree["children"]
        )
        
        # Check if search was successful
        max_value = max(
            node["value"] for node in tree["children"]
        )
        if max_value > 0.95:
            successful_searches += 1
            
        # Get path length
        best_child = max(
            tree["children"],
            key=lambda x: x["value"]
        )
        avg_path_length += len(best_child["reasoning_path"])
    
    metrics = {
        "avg_iterations": total_iterations / len(trees),
        "success_rate": successful_searches / len(trees),
        "avg_path_length": avg_path_length / len(trees)
    }
    
    return metrics

def compute_reasoning_accuracy(
    predictions: List[List[str]],
    targets: List[List[str]]
) -> float:
    """Compute accuracy of reasoning paths"""
    correct = 0
    total = len(predictions)
    
    for pred, target in zip(predictions, targets):
        if len(pred) == len(target):
            if all(p == t for p, t in zip(pred, target)):
                correct += 1
                
    return correct / total

def compute_reflection_metrics(
    reflection_paths: List[List[str]],
    targets: List[List[str]]
) -> Dict[str, float]:
    """Compute metrics for reflection paths"""
    reflection_count = 0
    successful_reflections = 0
    
    for reflection, target in zip(reflection_paths, targets):
        # Count reflections
        reflection_markers = [
            i for i, step in enumerate(reflection)
            if "incorrect" in step.lower()
        ]
        reflection_count += len(reflection_markers)
        
        # Check if reflection led to correct path
        for marker in reflection_markers:
            if marker + 2 < len(reflection):
                if reflection[marker + 2] in target:
                    successful_reflections += 1
    
    metrics = {
        "reflection_rate": reflection_count / len(reflection_paths),
        "reflection_success": (
            successful_reflections / reflection_count
            if reflection_count > 0 else 0.0
        )
    }
    
    return metrics
