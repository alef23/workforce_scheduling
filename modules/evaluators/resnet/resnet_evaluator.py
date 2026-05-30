from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# Bloque residual
# ============================================================

class ResidualBlock(nn.Module):
    """
    Bloque residual estándar:

        x -> Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN
          -> suma residual -> ReLU

    Mantiene:
        (B, channels, 28, 28)
    """

    def __init__(
        self,
        channels: int = 128,
    ) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )

        self.bn1 = nn.BatchNorm2d(channels)

        self.conv2 = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )

        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + residual
        out = F.relu(out)

        return out


# ============================================================
# ResNet evaluadora
# ============================================================

class WorkforceResNet(nn.Module):
    """
    Red evaluadora tipo ResNet para Workforce Scheduling.

    Input:
        x: torch.Tensor shape (B, 93, 28, 28)

    Outputs:
        policy_logits: torch.Tensor shape (B, 55)
        value: torch.Tensor shape (B,)

    Nota:
        Para entrenamiento es preferible devolver logits de política y aplicar
        CrossEntropy/KL/CrossEntropy soft manual sobre log_softmax.
        Para inferencia/MCTS, se puede convertir a probabilidades con softmax.
    """

    def __init__(
        self,
        input_channels: int = 93,
        board_height: int = 28,
        board_width: int = 28,
        hidden_channels: int = 128,
        num_res_blocks: int = 8,
        action_space_size: int = 55,
        policy_channels: int = 8,
        value_channels: int = 4,
        value_hidden_dim: int = 256,
    ) -> None:
        super().__init__()

        self.input_channels = input_channels
        self.board_height = board_height
        self.board_width = board_width
        self.hidden_channels = hidden_channels
        self.num_res_blocks = num_res_blocks
        self.action_space_size = action_space_size

        # ------------------------------------------------------------
        # Stem convolucional
        # ------------------------------------------------------------

        self.stem_conv = nn.Conv2d(
            in_channels=input_channels,
            out_channels=hidden_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )

        self.stem_bn = nn.BatchNorm2d(hidden_channels)

        # ------------------------------------------------------------
        # Backbone residual
        # ------------------------------------------------------------

        self.residual_blocks = nn.Sequential(
            *[
                ResidualBlock(channels=hidden_channels)
                for _ in range(num_res_blocks)
            ]
        )

        # ------------------------------------------------------------
        # Head de política
        # ------------------------------------------------------------

        self.policy_conv = nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=policy_channels,
            kernel_size=1,
            padding=0,
            bias=True,
        )

        self.policy_fc = nn.Linear(
            in_features=policy_channels * board_height * board_width,
            out_features=action_space_size,
        )

        # ------------------------------------------------------------
        # Head de valor
        # ------------------------------------------------------------

        self.value_conv = nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=value_channels,
            kernel_size=1,
            padding=0,
            bias=True,
        )

        self.value_fc1 = nn.Linear(
            in_features=value_channels * board_height * board_width,
            out_features=value_hidden_dim,
        )

        self.value_fc2 = nn.Linear(
            in_features=value_hidden_dim,
            out_features=1,
        )

        # Inicialización explícita de pesos
        self.initialize_weights()

    def initialize_weights(self) -> None:
        """
        Inicializa los pesos de la red.
    
        Criterio:
        - Conv2d: Kaiming normal para capas con ReLU.
        - BatchNorm2d: gamma=1, beta=0.
        - Linear: Xavier uniform.
        - Última capa de policy: pesos pequeños para logits iniciales cercanos a 0.
        - Última capa de value: pesos pequeños para valores iniciales cercanos a 0.
        """
    
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
    
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
    
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
        # Capas finales con escala chica
        nn.init.normal_(self.policy_fc.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.policy_fc.bias)
    
        nn.init.normal_(self.value_fc2.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.value_fc2.bias)

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Ejecuta la red.

        Parameters
        ----------
        x:
            Tensor de entrada shape (B, 93, 28, 28).

        Returns
        -------
        policy_logits:
            Tensor shape (B, 55). No tiene softmax aplicado.

        value:
            Tensor shape (B,), acotado en [-1, 1].
        """

        # Stem
        out = self.stem_conv(x)
        out = self.stem_bn(out)
        out = F.relu(out)

        # Residual backbone
        out = self.residual_blocks(out)

        # Policy head
        policy = self.policy_conv(out)
        policy = F.relu(policy)
        policy = torch.flatten(policy, start_dim=1)
        policy_logits = self.policy_fc(policy)

        # Value head
        value = self.value_conv(out)
        value = F.relu(value)
        value = torch.flatten(value, start_dim=1)
        value = self.value_fc1(value)
        value = F.relu(value)
        value = self.value_fc2(value)
        value = torch.tanh(value)
        value = value.squeeze(-1)

        return policy_logits, value

    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Método auxiliar para inferencia.

        Devuelve:
            policy_probs: softmax(policy_logits)
            value: valor escalar en [-1, 1]
        """

        self.eval()

        policy_logits, value = self.forward(x)
        policy_probs = F.softmax(policy_logits, dim=1)

        return policy_probs, value