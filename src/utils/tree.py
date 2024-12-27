from typing import Dict, List, Any
import torch

class ReasoningNode:
    def __init__(
        self,
        reasoning_path: List[str],
        value: float = 0.0,
        visits: int = 0,
        parent: Any = None
    ):
        self.reasoning_path = reasoning_path
        self.value = value
        self.visits = visits
        self.parent = parent
        self.children = []

    def add_child(self, child: 'ReasoningNode'):
        self.children.append(child)
        child.parent = self

def build_reasoning_tree(root_node: Dict) -> ReasoningNode:
    """Convert dictionary tree to ReasoningNode tree"""
    node = ReasoningNode(
        root_node["reasoning_path"],
        root_node["value"],
        root_node["visits"]
    )
    
    for child in root_node["children"]:
        child_node = build_reasoning_tree(child)
        node.add_child(child_node)
        
    return node

def get_reasoning_path(
    tree: ReasoningNode,
    target_value: float
) -> List[str]:
    """Extract reasoning path from tree with target value"""
    if not tree.children:
        return tree.reasoning_path if tree.value >= target_value else []
    
    for child in sorted(
        tree.children,
        key=lambda x: x.value,
        reverse=True
    ):
        path = get_reasoning_path(child, target_value)
        if path:
            return tree.reasoning_path + path
            
    return []

def print_tree(
    node: ReasoningNode,
    level: int = 0,
    max_depth: int = None
) -> None:
    """Visualize reasoning tree"""
    if max_depth is not None and level > max_depth:
        return
        
    indent = "  " * level
    print(f"{indent}Path: {node.reasoning_path}")
    print(f"{indent}Value: {node.value:.3f}")
    print(f"{indent}Visits: {node.visits}")
    
    for child in node.children:
        print_tree(child, level + 1, max_depth)
