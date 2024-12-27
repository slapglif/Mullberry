import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional
import pytorch_lightning as pl

class MulberryDataset(Dataset):
    def __init__(
        self,
        questions: List[str],
        labels: List[str],
        images: Optional[List[torch.Tensor]] = None
    ):
        self.questions = questions
        self.labels = labels
        self.images = images

    def __len__(self) -> int:
        return len(self.questions)

    def __getitem__(self, idx: int) -> Dict:
        item = {
            "questions": self.questions[idx],
            "labels": self.labels[idx]
        }
        
        if self.images is not None:
            item["images"] = self.images[idx]
            
        return item

class MulberryDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_questions: List[str],
        train_labels: List[str],
        val_questions: List[str],
        val_labels: List[str],
        train_images: Optional[List[torch.Tensor]] = None,
        val_images: Optional[List[torch.Tensor]] = None,
        batch_size: int = 32,
        num_workers: int = 4
    ):
        super().__init__()
        self.train_questions = train_questions
        self.train_labels = train_labels
        self.val_questions = val_questions
        self.val_labels = val_labels
        self.train_images = train_images
        self.val_images = val_images
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.train_dataset = MulberryDataset(
                self.train_questions,
                self.train_labels,
                self.train_images
            )
            
            self.val_dataset = MulberryDataset(
                self.val_questions,
                self.val_labels,
                self.val_images
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
