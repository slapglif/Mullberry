import pytest
import torch
import torch.nn as nn
from src.models.mulberry import Mulberry
from src.models.policy import PolicyModel, ReflectivePolicy
from src.data.dataset import MulberryDataset, MulberryDataModule
from src.training.trainer import train_mulberry
from src.models.comcts import CoMCTS
from typing import List, Optional, Sequence, Dict, Any

class MockPolicyModel(PolicyModel):
    def __init__(self):
        super().__init__()
        # Create proper mock layers with gradients
        self.embedding = nn.Linear(768, 128)
        self.hidden = nn.Linear(128, 128)
        self.output = nn.Linear(128, 64)  # Reduced vocab size for testing
        self.criterion = nn.CrossEntropyLoss()

    def forward(
        self,
        question: str,
        current_path: List[str],
        images: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        # Generate mock logits with gradients
        batch_size = 1 if isinstance(question, str) else len(question)
        mock_input = torch.randn(batch_size, 768, requires_grad=True)

        # Forward through layers
        hidden = self.embedding(mock_input)
        hidden = torch.relu(self.hidden(hidden))
        # Shape: [batch_size, seq_len=5, vocab_size=64]
        logits = self.output(hidden).unsqueeze(1).expand(-1, 5, -1)

        return {
            "logits": logits,
            "batch_size": batch_size
        }

    def evaluate(
        self,
        question: str,
        reasoning_path: List[str],
        images: Optional[torch.Tensor] = None
    ) -> float:
        # Generate mock evaluation score with gradients
        mock_input = torch.randn(1, 768, requires_grad=True)
        hidden = torch.relu(self.embedding(mock_input))
        hidden = torch.relu(self.hidden(hidden))
        score = torch.sigmoid(self.output(hidden).mean())
        return score.item()

@pytest.fixture
def mock_data():
    questions = ["What is 2+2?", "Explain this image"]
    labels = ["The answer is 4", "The image shows a cat"]
    return questions, labels

@pytest.fixture
def mock_model():
    base_model = MockPolicyModel()
    policy_models: Sequence[PolicyModel] = [MockPolicyModel() for _ in range(2)]
    return Mulberry(base_model, list(policy_models))

def test_dataset_creation(mock_data):
    questions, labels = mock_data
    dataset = MulberryDataset(questions, labels)

    assert len(dataset) == 2
    item = dataset[0]
    assert "questions" in item
    assert "labels" in item

def test_datamodule_creation(mock_data):
    questions, labels = mock_data
    datamodule = MulberryDataModule(
        questions, labels,
        questions, labels,
        batch_size=1
    )

    datamodule.setup()
    assert datamodule.train_dataloader() is not None
    assert datamodule.val_dataloader() is not None

def test_model_initialization(mock_model):
    assert isinstance(mock_model.base_model, PolicyModel)
    assert len(mock_model.comcts.policy_models) == 2
    assert mock_model.training

def test_comcts_operations():
    policy_models: Sequence[PolicyModel] = [MockPolicyModel() for _ in range(2)]
    comcts = CoMCTS(list(policy_models))

    # Test node expansion
    init_node = {
        "reasoning_path": [],
        "value": 0.0,
        "visits": 0,
        "children": [],
        "parent": None
    }

    candidates = comcts.expand_nodes(init_node, "What is 2+2?")
    assert len(candidates) == 2

    # Test simulation and evaluation
    valid_candidates = comcts.simulate_and_evaluate(
        candidates,
        "What is 2+2?"
    )
    assert len(valid_candidates) > 0

    # Test node selection
    selected = comcts.select_node(valid_candidates)
    assert selected is not None

def test_training_loop(mock_model, mock_data, tmp_path):
    questions, labels = mock_data
    datamodule = MulberryDataModule(
        questions, labels,
        questions, labels,
        batch_size=1,
        num_workers=0  # Use single worker for testing
    )

    checkpoint_dir = str(tmp_path / "checkpoints")
    tensorboard_dir = str(tmp_path / "logs")

    try:
        train_mulberry(
            mock_model,
            datamodule,
            max_epochs=1,
            checkpoint_dir=checkpoint_dir,
            tensorboard_dir=tensorboard_dir
        )
        training_completed = True
    except Exception as e:
        print(f"Training failed with error: {str(e)}")
        training_completed = False

    assert training_completed, "Training should complete without errors"

def test_reflective_policy():
    base_model = MockPolicyModel()
    reflection_head = nn.Linear(128, 1)  # Match hidden size
    policy = ReflectivePolicy(base_model, reflection_head)

    # Test inference mode
    output = policy.predict_step("What is 2+2?", ["First step"])
    assert isinstance(output, List)
    assert len(output) > 0

    # Test training mode
    train_output = policy("What is 2+2?", ["First step"])
    assert isinstance(train_output, Dict)
    assert "logits" in train_output
    assert "batch_size" in train_output

    score = policy.evaluate("What is 2+2?", ["First step", "Second step"])
    assert isinstance(score, float)
    assert 0 <= score <= 1