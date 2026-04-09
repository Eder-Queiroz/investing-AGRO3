"""
Testes unitários para src/models/mlp.py.

Cobertura: 24 testes em 6 grupos.
  1. Constructor validation   — erros antecipados por argumentos inválidos
  2. Forward pass shape       — geometria de output para vários batch sizes
  3. Output semantics         — logits vs probabilidades, determinismo, NaN
  4. Architecture integrity   — contagem de camadas Linear, BN, Dropout
  5. Factory function         — build_mlp_from_config
  6. Gradient flow            — backpropagation alcança todos os parâmetros

Todos os testes rodam em CPU, sem dados reais e sem acoplamento ao Trainer.
Cada teste completa em milissegundos.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
import torch.nn as nn
from src.models.mlp import ValueInvestingMLP, build_mlp_from_config

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_config() -> dict[str, Any]:
    """Config mínimo válido para testes rápidos."""
    return {
        "model": {
            "hidden_layers": [16, 8],
            "dropout_rates": [0.1, 0.1],
            "num_classes": 3,
            "use_batch_norm": True,
            "activation": "relu",
        }
    }


@pytest.fixture
def production_config() -> dict[str, Any]:
    """Config de produção conforme model_config.yaml."""
    return {
        "model": {
            "hidden_layers": [512, 256, 128, 64],
            "dropout_rates": [0.3, 0.3, 0.2, 0.2],
            "num_classes": 3,
            "use_batch_norm": True,
            "activation": "relu",
        }
    }


@pytest.fixture
def small_model() -> ValueInvestingMLP:
    """Modelo pequeno: input=20, hidden=[16,8] — para testes rápidos."""
    torch.manual_seed(0)
    return ValueInvestingMLP(
        input_dim=20,
        hidden_layers=[16, 8],
        dropout_rates=[0.1, 0.1],
        num_classes=3,
        use_batch_norm=True,
        activation="relu",
    )


@pytest.fixture
def production_model() -> ValueInvestingMLP:
    """Modelo de produção: input=1196, conforme CLAUDE.md."""
    torch.manual_seed(0)
    return ValueInvestingMLP(
        input_dim=1196,
        hidden_layers=[512, 256, 128, 64],
        dropout_rates=[0.3, 0.3, 0.2, 0.2],
        num_classes=3,
        use_batch_norm=True,
        activation="relu",
    )


# ---------------------------------------------------------------------------
# Grupo 1: Constructor validation (6 testes)
# ---------------------------------------------------------------------------


def test_raises_on_mismatched_hidden_dropout_lengths() -> None:
    """hidden_layers e dropout_rates devem ter o mesmo comprimento."""
    with pytest.raises(ValueError, match="mesmo comprimento"):
        ValueInvestingMLP(
            input_dim=20,
            hidden_layers=[16, 8],
            dropout_rates=[0.1],  # faltando um elemento
            num_classes=3,
            use_batch_norm=True,
            activation="relu",
        )


def test_raises_on_zero_input_dim() -> None:
    """input_dim=0 deve falhar com ValueError."""
    with pytest.raises(ValueError, match="input_dim"):
        ValueInvestingMLP(
            input_dim=0,
            hidden_layers=[16],
            dropout_rates=[0.1],
            num_classes=3,
            use_batch_norm=True,
            activation="relu",
        )


def test_raises_on_negative_input_dim() -> None:
    """input_dim negativo deve falhar com ValueError."""
    with pytest.raises(ValueError, match="input_dim"):
        ValueInvestingMLP(
            input_dim=-5,
            hidden_layers=[16],
            dropout_rates=[0.1],
            num_classes=3,
            use_batch_norm=True,
            activation="relu",
        )


def test_raises_on_invalid_activation_name() -> None:
    """Ativação desconhecida deve falhar com ValueError."""
    with pytest.raises(ValueError, match="não suportada"):
        ValueInvestingMLP(
            input_dim=20,
            hidden_layers=[16],
            dropout_rates=[0.1],
            num_classes=3,
            use_batch_norm=True,
            activation="sigmoid",  # não está em _ACTIVATIONS
        )


def test_raises_on_dropout_rate_gte_one() -> None:
    """dropout_rate >= 1.0 deve falhar com ValueError."""
    with pytest.raises(ValueError, match="dropout_rates"):
        ValueInvestingMLP(
            input_dim=20,
            hidden_layers=[16],
            dropout_rates=[1.0],  # limite inválido
            num_classes=3,
            use_batch_norm=True,
            activation="relu",
        )


def test_raises_on_num_classes_less_than_two() -> None:
    """num_classes < 2 é semanticamente inválido."""
    with pytest.raises(ValueError, match="num_classes"):
        ValueInvestingMLP(
            input_dim=20,
            hidden_layers=[16],
            dropout_rates=[0.1],
            num_classes=1,
            use_batch_norm=True,
            activation="relu",
        )


# ---------------------------------------------------------------------------
# Grupo 2: Forward pass shape (5 testes)
# ---------------------------------------------------------------------------


def test_forward_output_shape_single_sample(small_model: ValueInvestingMLP) -> None:
    """Input (1, 20) deve produzir output (1, 3)."""
    small_model.eval()
    x = torch.randn(1, 20)
    out = small_model(x)
    assert out.shape == (1, 3)


def test_forward_output_shape_batch(small_model: ValueInvestingMLP) -> None:
    """Input (64, 20) deve produzir output (64, 3)."""
    small_model.eval()
    x = torch.randn(64, 20)
    out = small_model(x)
    assert out.shape == (64, 3)


def test_forward_output_shape_production_model(
    production_model: ValueInvestingMLP,
) -> None:
    """Modelo de produção: input (4, 1196) deve produzir (4, 3)."""
    production_model.eval()
    x = torch.randn(4, 1196)
    out = production_model(x)
    assert out.shape == (4, 3)


def test_forward_output_dtype_is_float32(small_model: ValueInvestingMLP) -> None:
    """Output deve ser float32 (input também é float32)."""
    small_model.eval()
    x = torch.randn(4, 20)  # randn é float32 por padrão
    out = small_model(x)
    assert out.dtype == torch.float32


def test_forward_handles_batch_size_one_in_eval_mode(
    small_model: ValueInvestingMLP,
) -> None:
    """BatchNorm com batch_size=1 em modo eval() usa running stats — deve funcionar.

    Em modo train(), batch_size=1 causa erro no BatchNorm1d (variância não definida).
    Este teste captura o bug de produção onde o último batch pode ter size=1.
    """
    small_model.eval()  # crítico: eval() não train()
    x = torch.randn(1, 20)
    out = small_model(x)
    assert out.shape == (1, 3)
    assert not torch.isnan(out).any()


# ---------------------------------------------------------------------------
# Grupo 3: Output semantics (4 testes)
# ---------------------------------------------------------------------------


def test_output_is_logits_not_probabilities(small_model: ValueInvestingMLP) -> None:
    """Output é logits brutos — NOT deve somar a 1.0 por linha."""
    small_model.eval()
    x = torch.randn(8, 20)
    logits = small_model(x)
    row_sums = logits.sum(dim=1)
    # Logits arbitrários: sum ≠ 1.0
    # Softmax(logits) deve somar a 1.0
    probs = torch.softmax(logits, dim=1)
    assert not torch.allclose(row_sums, torch.ones(8), atol=1e-3), (
        "Se somas são ~1.0, o modelo pode estar aplicando softmax desnecessário"
    )
    assert torch.allclose(probs.sum(dim=1), torch.ones(8), atol=1e-5)


def test_train_mode_is_stochastic_eval_mode_is_deterministic(
    small_model: ValueInvestingMLP,
) -> None:
    """Dropout em train() introduz estocasticidade; eval() é determinístico."""
    torch.manual_seed(42)
    x = torch.randn(8, 20)

    # Em train(): duas passagens devem diferir (Dropout ativo)
    small_model.train()
    out1 = small_model(x)
    out2 = small_model(x)
    assert not torch.allclose(out1, out2), (
        "Modelo em train() deve ser estocástico (Dropout)"
    )

    # Em eval(): duas passagens devem ser idênticas (Dropout desativado)
    small_model.eval()
    out3 = small_model(x)
    out4 = small_model(x)
    assert torch.allclose(out3, out4), (
        "Modelo em eval() deve ser determinístico"
    )


def test_no_nan_in_output_with_valid_input(small_model: ValueInvestingMLP) -> None:
    """Input normal não deve produzir NaN no output."""
    small_model.eval()
    x = torch.randn(16, 20)
    out = small_model(x)
    assert not torch.isnan(out).any(), "Output contém NaN para input normal"


def test_no_nan_in_output_with_extreme_inputs(small_model: ValueInvestingMLP) -> None:
    """Input com valores extremos (±100) não deve explodir para NaN com BatchNorm."""
    small_model.eval()
    x_pos = torch.full((4, 20), fill_value=100.0)
    x_neg = torch.full((4, 20), fill_value=-100.0)
    out_pos = small_model(x_pos)
    out_neg = small_model(x_neg)
    assert not torch.isnan(out_pos).any(), "NaN com input=+100"
    assert not torch.isnan(out_neg).any(), "NaN com input=-100"


# ---------------------------------------------------------------------------
# Grupo 4: Architecture integrity (4 testes)
# ---------------------------------------------------------------------------


def test_correct_number_of_linear_layers(small_model: ValueInvestingMLP) -> None:
    """Deve haver len(hidden_layers) + 1 camadas Linear (hidden + output)."""
    linear_count = sum(1 for m in small_model.modules() if isinstance(m, nn.Linear))
    # small_model tem hidden=[16, 8] → 2 hidden + 1 output = 3
    assert linear_count == 3


def test_batch_norm_layers_present_when_enabled() -> None:
    """Com use_batch_norm=True, deve haver len(hidden_layers) camadas BN."""
    model = ValueInvestingMLP(
        input_dim=20,
        hidden_layers=[32, 16, 8],
        dropout_rates=[0.1, 0.1, 0.1],
        num_classes=3,
        use_batch_norm=True,
        activation="relu",
    )
    bn_count = sum(1 for m in model.modules() if isinstance(m, nn.BatchNorm1d))
    assert bn_count == 3  # um por camada oculta


def test_no_batch_norm_when_disabled() -> None:
    """Com use_batch_norm=False, não deve haver nenhuma camada BatchNorm1d."""
    model = ValueInvestingMLP(
        input_dim=20,
        hidden_layers=[16, 8],
        dropout_rates=[0.1, 0.1],
        num_classes=3,
        use_batch_norm=False,
        activation="relu",
    )
    bn_count = sum(1 for m in model.modules() if isinstance(m, nn.BatchNorm1d))
    assert bn_count == 0


def test_dropout_layers_present(small_model: ValueInvestingMLP) -> None:
    """Deve haver len(hidden_layers) camadas Dropout."""
    dropout_count = sum(1 for m in small_model.modules() if isinstance(m, nn.Dropout))
    # small_model: hidden=[16,8] → 2 dropouts
    assert dropout_count == 2


# ---------------------------------------------------------------------------
# Grupo 5: Factory function — build_mlp_from_config (3 testes)
# ---------------------------------------------------------------------------


def test_factory_returns_value_investing_mlp_instance(
    small_config: dict[str, Any],
) -> None:
    """build_mlp_from_config deve retornar uma instância de ValueInvestingMLP."""
    model = build_mlp_from_config(small_config, input_dim=20)
    assert isinstance(model, ValueInvestingMLP)


def test_factory_respects_input_dim(small_config: dict[str, Any]) -> None:
    """Modelo construído pela fábrica deve aceitar tensor com input_dim correto."""
    model = build_mlp_from_config(small_config, input_dim=20)
    model.eval()
    x = torch.randn(4, 20)
    out = model(x)
    assert out.shape == (4, 3)


def test_factory_raises_on_missing_config_key(small_config: dict[str, Any]) -> None:
    """Config sem chave obrigatória deve levantar KeyError."""
    del small_config["model"]["hidden_layers"]
    with pytest.raises(KeyError):
        build_mlp_from_config(small_config, input_dim=20)


# ---------------------------------------------------------------------------
# Grupo 6: Gradient flow (2 testes)
# ---------------------------------------------------------------------------


def test_gradients_flow_to_all_parameters(small_model: ValueInvestingMLP) -> None:
    """Todos os parâmetros treináveis devem receber gradiente não-nulo após backward.

    Detecta neurônios mortos, camadas desconectadas ou inicialização errada.
    """
    torch.manual_seed(0)
    small_model.train()
    x = torch.randn(8, 20)
    y = torch.randint(0, 3, (8,))

    logits = small_model(x)
    loss = nn.CrossEntropyLoss()(logits, y)
    loss.backward()

    for name, param in small_model.named_parameters():
        assert param.grad is not None, f"Parâmetro '{name}' não recebeu gradiente"
        assert param.grad.abs().sum().item() > 0, (
            f"Gradiente de '{name}' é zero (possível neurônio morto)"
        )


def test_gradient_clipping_reduces_norm() -> None:
    """clip_grad_norm_ deve reduzir a norma total para ≤ max_norm.

    Simula o que o Trainer faz após cada backward pass.
    """
    torch.manual_seed(0)
    model = ValueInvestingMLP(
        input_dim=20,
        hidden_layers=[16, 8],
        dropout_rates=[0.0, 0.0],  # sem dropout para resultado determinístico
        num_classes=3,
        use_batch_norm=False,
        activation="relu",
    )
    model.train()

    x = torch.randn(4, 20)
    y = torch.randint(0, 3, (4,))
    logits = model(x)
    loss = nn.CrossEntropyLoss()(logits, y)
    loss.backward()

    # Força gradiente grande em um parâmetro
    first_param = next(iter(model.parameters()))
    first_param.grad = torch.full_like(first_param.grad, fill_value=1000.0)

    max_norm = 1.0
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)

    total_norm = torch.sqrt(
        sum(p.grad.norm() ** 2 for p in model.parameters() if p.grad is not None)
    ).item()

    assert total_norm <= max_norm + 1e-3, (
        f"Norma após clipping ({total_norm:.4f}) excede max_norm ({max_norm})"
    )
