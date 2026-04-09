"""
ValueInvestingMLP — Fase 4: Arquitetura MLP para classificação de sinais AGRO3.

## Responsabilidade

Define a arquitetura MLP (Multi-Layer Perceptron) usada para classificar
janelas deslizantes de features financeiras em três sinais de decisão:
    0 → SELL  (Vender)
    1 → HOLD  (Aguardar)
    2 → BUY   (Comprar)

## Decisões de Design

- **Rede como nn.Sequential único**: toda a computação está em `self.network`,
  tornando `forward()` trivial e eliminando bugs de encadeamento manual.

- **BatchNorm antes da ativação**: estabiliza gradientes para séries financeiras
  com alta variância entre features (preços, taxas, z-scores coexistem).

- **bias=not use_batch_norm**: quando BatchNorm está ativo, seu parâmetro β
  absorve o viés do Linear. Manter ambos desperdiça parâmetros.

- **Inicialização He (Kaiming)**: adequada para ativações ReLU e variantes;
  mantém variância dos gradientes estável à medida que a profundidade aumenta.

- **Output = logits**: a camada de saída não aplica softmax. CrossEntropyLoss
  espera logits não normalizados; aplicar softmax antes causaria instabilidade.

## Uso

    from src.models.mlp import build_mlp_from_config
    from src.utils.config import load_model_config

    cfg = load_model_config()
    model = build_mlp_from_config(cfg, input_dim=1196)
    logits = model(x_batch)  # shape: (batch_size, 3)
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

from src.utils.logger import get_logger

logger: logging.Logger = get_logger(__name__)

# Ativações suportadas — extensível sem modificar o construtor
_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "gelu": nn.GELU,
}


class ValueInvestingMLP(nn.Module):
    """MLP para classificação de sinais de Value Investing na AGRO3.

    Recebe um vetor achatado de janela deslizante (W semanas × F features)
    e produz logits para 3 classes: SELL, HOLD, BUY.

    Arquitetura por camada oculta `i`:
        Linear(in, hidden[i], bias=not use_batch_norm)
        BatchNorm1d(hidden[i])  ← se use_batch_norm
        Activation()
        Dropout(dropout_rates[i])

    Camada de saída:
        Linear(hidden[-1], num_classes)  ← sem BN/dropout/ativação

    Atributos:
        network: nn.Sequential completo com todas as camadas.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_layers: list[int],
        dropout_rates: list[float],
        num_classes: int,
        use_batch_norm: bool,
        activation: str,
    ) -> None:
        """Constrói a MLP e aplica inicialização He.

        Args:
            input_dim: Número de neurônios de entrada (W * F, ex: 1196).
            hidden_layers: Lista com número de neurônios por camada oculta.
            dropout_rates: Taxa de dropout por camada oculta (mesmo comprimento
                que hidden_layers).
            num_classes: Número de classes de output (3: SELL/HOLD/BUY).
            use_batch_norm: Se True, insere BatchNorm1d após cada Linear oculto.
            activation: Nome da função de ativação ('relu', 'leaky_relu', 'gelu').

        Raises:
            ValueError: Se os argumentos violarem as invariantes de design.
        """
        super().__init__()

        # --- Validação de argumentos ---
        if len(hidden_layers) != len(dropout_rates):
            raise ValueError(
                f"hidden_layers e dropout_rates devem ter o mesmo comprimento. "
                f"Recebido: len(hidden_layers)={len(hidden_layers)}, "
                f"len(dropout_rates)={len(dropout_rates)}"
            )
        if input_dim <= 0:
            raise ValueError(f"input_dim deve ser > 0. Recebido: {input_dim}")
        if num_classes < 2:
            raise ValueError(f"num_classes deve ser >= 2. Recebido: {num_classes}")
        if any(r >= 1.0 or r < 0.0 for r in dropout_rates):
            raise ValueError(
                f"Todos os dropout_rates devem estar em [0, 1). Recebido: {dropout_rates}"
            )
        if activation not in _ACTIVATIONS:
            raise ValueError(
                f"Ativação '{activation}' não suportada. Opções: {sorted(_ACTIVATIONS.keys())}"
            )

        activation_cls = _ACTIVATIONS[activation]

        # --- Construção do Sequential ---
        layers: list[nn.Module] = []
        in_features = input_dim

        for hidden_size, dropout_rate in zip(hidden_layers, dropout_rates, strict=True):
            # Linear: sem bias quando BatchNorm está ativo (β do BN absorve o viés)
            layers.append(nn.Linear(in_features, hidden_size, bias=not use_batch_norm))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(activation_cls())
            layers.append(nn.Dropout(p=dropout_rate))
            in_features = hidden_size

        # Camada de saída: logits puros, sem BN/dropout/ativação
        layers.append(nn.Linear(in_features, num_classes, bias=True))

        self.network: nn.Sequential = nn.Sequential(*layers)

        # --- Inicialização He (Kaiming) ---
        self._init_weights()

        n_params = sum(p.numel() for p in self.parameters())
        logger.info(
            f"ValueInvestingMLP construída: "
            f"input={input_dim} → hidden={hidden_layers} → output={num_classes} | "
            f"BatchNorm={use_batch_norm} | activation={activation} | "
            f"params={n_params:,}"
        )

    def _init_weights(self) -> None:
        """Aplica inicialização He uniforme a todas as camadas Linear."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Passagem direta pela rede.

        Args:
            x: Tensor float32 de shape (batch_size, input_dim).

        Returns:
            Logits de shape (batch_size, num_classes). NÃO aplicar softmax
            antes de CrossEntropyLoss — ela aplica log_softmax internamente.
        """
        return self.network(x)


def build_mlp_from_config(
    config: dict[str, Any],
    input_dim: int,
) -> ValueInvestingMLP:
    """Fábrica que constrói ValueInvestingMLP a partir do model_config.yaml.

    `input_dim` é passado explicitamente (não lido do config) porque é uma
    propriedade dos dados em tempo de execução (W × F), não um hiperparâmetro
    de arquitetura independente.

    Args:
        config: Dicionário completo do model_config.yaml (ou sub-dict 'model').
        input_dim: Dimensão do vetor de entrada — deve ser W * num_features.

    Returns:
        Instância de ValueInvestingMLP configurada e pronta para treinamento.

    Raises:
        KeyError: Se chaves obrigatórias estiverem ausentes do config.
        ValueError: Se os valores violarem invariantes de design da MLP.
    """
    # Suporta tanto config completo quanto sub-dict 'model'
    model_cfg: dict[str, Any] = config.get("model", config)

    return ValueInvestingMLP(
        input_dim=input_dim,
        hidden_layers=model_cfg["hidden_layers"],
        dropout_rates=model_cfg["dropout_rates"],
        num_classes=model_cfg["num_classes"],
        use_batch_norm=model_cfg["use_batch_norm"],
        activation=model_cfg["activation"],
    )
