import torch
import torch.nn as nn
import torchvision.models as models

class FusionModel(nn.Module):
    def __init__(self, audio_features=102, num_classes=8):
        super(FusionModel, self).__init__()

        # ResNet18 backbone
        base_model = models.resnet18(pretrained=True)
        base_model.fc = nn.Identity()  # Remove final classification layer
        self.cnn = base_model
        self.cnn_fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        # Audio MLP branch
        self.audio_branch = nn.Sequential(
            nn.Linear(audio_features, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.LayerNorm(128)
        )

        # Fusion MLP
        self.fusion_fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, image, audio):
        # Extract visual features
        image_feat = self.cnn(image)
        image_feat = self.cnn_fc(image_feat)

        # Extract audio features
        audio_feat = self.audio_branch(audio)

        # Combine features
        fused = torch.cat((image_feat, audio_feat), dim=1)

        # Predict
        out = self.fusion_fc(fused)
        return out
