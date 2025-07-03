# File: Core/model.py

from abc import ABC
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions.categorical import Categorical

def conv_shape(input, kernel_size, stride, padding=0):
    return (input + 2 * padding - kernel_size) // stride + 1


# === Policy Network (unchanged) ===
class PolicyModel(nn.Module, ABC):
    def __init__(self, state_shape, n_actions):
        super(PolicyModel, self).__init__()
        self.state_shape = state_shape
        self.n_actions = n_actions

        c, w, h = state_shape
        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        flatten_size = 32 * 7 * 7

        self.fc1 = nn.Linear(flatten_size, 256)
        self.gru = nn.GRUCell(256, 256)

        self.extra_value_fc = nn.Linear(256, 256)
        self.extra_policy_fc = nn.Linear(256, 256)

        self.policy = nn.Linear(256, self.n_actions)
        self.int_value = nn.Linear(256, 1)
        self.ext_value = nn.Linear(256, 1)

        # Orthogonal initialization
        for layer in self.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                if layer.bias is not None:
                    layer.bias.data.zero_()

    def forward(self, inputs, hidden_state):
        if inputs.ndim == 5:
            inputs = inputs.squeeze(1)

        x = inputs / 255.0
        x = self.conv(x)
        x = F.relu(self.fc1(x))
        h = self.gru(x, hidden_state)

        x_v = h + F.relu(self.extra_value_fc(h))
        x_pi = h + F.relu(self.extra_policy_fc(h))

        int_value = self.int_value(x_v)
        ext_value = self.ext_value(x_v)

        policy_logits = self.policy(x_pi)
        probs = F.softmax(policy_logits, dim=1)
        dist = Categorical(probs)

        return dist, int_value, ext_value, probs, h


# === TargetModel: a frozen 3-conv → 512-dim network ===
class TargetModel(nn.Module, ABC):
    def __init__(self, state_shape):
        super(TargetModel, self).__init__()
        c, w, h = state_shape

        self.conv1 = nn.Conv2d(c, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        # After three convs (7×7 → 7×7 each), we have 64×7×7 = 64*49 = 3136 features
        flatten_size = 64 * 7 * 7

        self.encoded_features = nn.Linear(flatten_size, 512)

        self._init_weights()

    def _init_weights(self):
        for layer in [self.conv1, self.conv2, self.conv3, self.encoded_features]:
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                if layer.bias is not None:
                    layer.bias.data.zero_()

    def forward(self, inputs):
        """
        inputs: [B, C, H, W], values in [0,255]
        """
        x = inputs / 255.0
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)        # → [B, flatten_size]
        feat = self.encoded_features(x)  # → [B, 512]
        return feat                      # frozen during training


# === PredictorModel: same convs + 2 FC layers → 512-dim ===
class PredictorModel(nn.Module, ABC):
    def __init__(self, state_shape):
        super(PredictorModel, self).__init__()
        c, w, h = state_shape

        self.conv1 = nn.Conv2d(c, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        flatten_size = 64 * 7 * 7

        self.fc_hidden = nn.Linear(flatten_size, 512)
        self.fc_pred = nn.Linear(512, 512)

        self._init_weights()

    def _init_weights(self):
        # Hidden layers: gain = √2
        for layer in [self.conv1, self.conv2, self.conv3, self.fc_hidden]:
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                if layer.bias is not None:
                    layer.bias.data.zero_()
        # Final output layer: smaller gain = √0.01
        if isinstance(self.fc_pred, nn.Linear):
            nn.init.orthogonal_(self.fc_pred.weight, gain=np.sqrt(0.01))
            if self.fc_pred.bias is not None:
                self.fc_pred.bias.data.zero_()

    def forward(self, inputs):
        """
        inputs: [B, C, H, W], values in [0,255]
        """
        x = inputs / 255.0
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)     # → [B, flatten_size]
        x = F.relu(self.fc_hidden(x)) # → [B, 512]
        pred = self.fc_pred(x)        # → [B, 512]
        return pred
