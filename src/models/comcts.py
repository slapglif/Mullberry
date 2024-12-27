import torch
import torch.nn as nn
import einops
import math
from typing import List, Dict, Tuple

class CoMCTS(nn.Module):
    def __init__(
        self,
        policy_models: List[nn.Module],
        max_search_iterations: int = 20,
        exploration_constant: float = 1.0,
        ucb_threshold: float = 0.0
    ):
        super().__init__()
        self.policy_models = policy_models
        self.max_search_iterations = max_search_iterations
        self.exploration_constant = exploration_constant
        self.ucb_threshold = ucb_threshold

    def expand_nodes(self, current_node: Dict, question: str) -> List[Dict]:
        """Joint expansion using collective knowledge from multiple models"""
        candidate_paths = []

        # Using ThreadPoolExecutor for parallel processing on CPU
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor() as executor:
            future_paths = [
                executor.submit(model, question, current_node["reasoning_path"])
                for model in self.policy_models
            ]

            for future in future_paths:
                path = future.result()
                candidate_paths.append({
                    "reasoning_path": path,
                    "value": 0.0,
                    "visits": 0,
                    "parent": current_node,
                    "children": []
                })

        return candidate_paths

    def simulate_and_evaluate(
        self, 
        candidates: List[Dict],
        question: str
    ) -> List[Dict]:
        """Joint simulation and error positioning"""
        valid_candidates = []

        # Parallel evaluation using ThreadPoolExecutor
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor() as executor:
            def evaluate_candidate(candidate):
                scores = []
                futures = [
                    executor.submit(
                        model.evaluate,
                        question,
                        candidate["reasoning_path"]
                    )
                    for model in self.policy_models
                ]

                for future in futures:
                    scores.append(future.result())

                avg_score = sum(scores) / len(scores)
                if avg_score >= self.ucb_threshold:
                    candidate["value"] = avg_score
                    return candidate
                return None

            results = executor.map(evaluate_candidate, candidates)
            valid_candidates = [r for r in results if r is not None]

        return valid_candidates

    def backpropagate(self, node: Dict, value: float):
        """Bottom-up update of statistics"""
        current = node
        while current:
            current["visits"] += 1
            # Update node value using children's values
            if current["children"]:
                child_values = [c["value"] for c in current["children"]]
                current["value"] = (
                    current["visits"] * current["value"] + sum(child_values)
                ) / (current["visits"] + len(current["children"]))
            current = current.get("parent", None)

    def select_node(self, candidates: List[Dict]) -> Dict:
        """Select node with highest UCB value"""
        if not candidates:
            return None

        # Vectorized UCB computation for efficiency
        values = torch.tensor([node["value"] for node in candidates])
        visits = torch.tensor([node["visits"] for node in candidates])
        parent_visits = torch.tensor(
            [node["parent"]["visits"] for node in candidates]
        )

        # UCB formula vectorized
        exploration = torch.sqrt(
            torch.log(parent_visits) / (1 + visits)
        )
        ucb_scores = values + self.exploration_constant * exploration

        best_idx = ucb_scores.argmax().item()
        return candidates[best_idx]

    def search(
        self,
        question: str,
        initial_node: Dict
    ) -> Tuple[List[str], Dict]:
        """Main CoMCTS search loop with CPU optimization"""
        root = initial_node
        best_path = None
        best_value = float("-inf")

        for _ in range(self.max_search_iterations):
            # Expansion
            candidates = self.expand_nodes(root, question)

            # Simulation & Error Positioning
            valid_candidates = self.simulate_and_evaluate(candidates, question)

            if not valid_candidates:
                continue

            # Backpropagation
            for candidate in valid_candidates:
                self.backpropagate(candidate, candidate["value"])
                root["children"].extend(valid_candidates)

                # Update best path if found better one
                if candidate["value"] > best_value:
                    best_value = candidate["value"]
                    best_path = candidate["reasoning_path"]

            # Selection
            root = self.select_node(valid_candidates)
            if root is None:
                break

            # Early stopping if we found a good path
            if best_value > 0.95:
                break

        return best_path, root

    def get_reflective_path(
        self,
        tree: Dict,
        positive_path: List[str]
    ) -> List[str]:
        """Find reflective reasoning path with error correction"""
        reflective_path = []
        current_node = tree

        for step in positive_path:
            # Find negative sibling with lowest UCB
            siblings = [
                c for c in current_node["children"] 
                if c["reasoning_path"][-1] != step
            ]

            if siblings:
                # Vectorized computation for finding negative sibling
                sibling_values = torch.tensor([s["value"] for s in siblings])
                neg_idx = sibling_values.argmin().item()
                neg_sibling = siblings[neg_idx]

                # Add reflection transition
                reflective_path.extend([
                    neg_sibling["reasoning_path"][-1],
                    "The previous reasoning step was incorrect. Let's correct it.",
                    step
                ])
            else:
                reflective_path.append(step)

            # Move to next node
            current_node = next(
                c for c in current_node["children"]
                if c["reasoning_path"][-1] == step
            )

        return reflective_path