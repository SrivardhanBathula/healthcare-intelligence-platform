import torch
import torch.nn as nn
from torchvision import models, transforms
from typing import List, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)


class MedicalImagingClassifier(nn.Module):
    """Fine-tuned DenseNet for medical image classification with GPU batch optimization."""

    def __init__(self, num_classes: int = 14, pretrained: bool = True):
        super().__init__()
        self.backbone = models.densenet121(pretrained=pretrained)
        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )
        self.num_classes = num_classes
        self.transform = transforms.Compose([
            transforms.Resize(256), transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    @torch.inference_mode()
    def batch_predict(self, images: List, batch_size: int = 64,
                     device: str = "cuda") -> np.ndarray:
        self.eval().to(device)
        all_probs = []
        for i in range(0, len(images), batch_size):
            batch = torch.stack([self.transform(img) for img in images[i:i+batch_size]]).to(device)
            with torch.amp.autocast("cuda"):
                logits = self(batch)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
        return np.vstack(all_probs)
