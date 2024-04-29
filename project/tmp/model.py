from transformers import ViTModel, ViTConfig
import torch.nn.functional as F
import torch.nn as nn
import torch


class AlphaVitModel(nn.Module):

    def __init__(
        self, input_size: int, path_size: int, latent_size: int, output_size: int
    ):
        super(AlphaVitModel, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.config = ViTConfig(
            hidden_size=latent_size,
            image_size=input_size,
            patch_size=path_size,
            num_channels=1,
        )

        self.vit = ViTModel(self.config).to(self.device)

        self.policy_head = nn.Sequential(
            nn.Linear(latent_size, 256),
            nn.ReLU(),
            nn.Linear(256, output_size),
        ).to(self.device)

        self.value_head = nn.Sequential(
            nn.Linear(latent_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        ).to(self.device)

    def forward(self, x: torch.Tensor) -> tuple:
        x = self.vit(x.to(self.device)).last_hidden_state[:, 0]
        x = x[None, None, :, :]
        policy = self.policy_head(x)
        value = self.value_head(x)
        return nn.Softmax(dim=-1)(policy), value
