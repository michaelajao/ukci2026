"""Per-region Physics-Informed Neural Network for SEI_aI_sHCRD epidemic modelling.

This module implements the per-region PINN-SEIRD forecaster described in
docs/02_METHODOLOGY.md §1. The compartmental structure is reused from
Ajao-Olarinoye et al. (2025) [book chapter] with one refinement:

    The 2025 chapter overloaded the parameter omega to represent both the
    I_s -> H hospitalisation ratio AND, via (1 - omega), the H -> C critical-
    care escalation. These are biologically distinct quantities. We separate
    them by introducing phi for the H -> C escalation, leaving omega for the
    I_s -> H hospitalisation ratio alone.

Two networks per region:
  * StateNet:    t -> (S, E, I_a, I_s, H, C, R, D)
  * ParameterNet: t -> (beta, gamma_c, delta_c, eta)

The PINN loss is the sum of:
  * Data fit on observable compartments (I_s, H, C, D)
  * ODE residual on the SEIRD system

Fixed parameters (alpha, rho, d_s, d_a, d_H, mu, omega, phi) are taken from
Ajao-Olarinoye et al. (2025) Table 1, with the refinement noted above.

Note on units: alpha, d_s, d_a, d_H are RATES (inverse-days). Their reciprocals
are clinical periods (e.g. 1/alpha = 5 days incubation period).

Usage (see ``forecasting.train_forecasters`` for the full training loop):

    from forecasting.pinn_seird import (
        SEIRDFixedParams, RegionalPINN, pinn_loss
    )
    fixed = SEIRDFixedParams()
    pinn = RegionalPINN(population=8_982_000)  # London population
    state, params = pinn(t)
    loss, parts = pinn_loss(pinn, t_data, y_data, t_collocation, fixed)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


# ---------------------------------------------------------------------------
# Fixed parameters from Ajao-Olarinoye et al. (2025) Table 1
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SEIRDFixedParams:
    """Compartment-flow parameters that are fixed (not learned).

    Refined from Ajao-Olarinoye et al. (2025) Table 1: we introduce a separate
    parameter ``phi`` for the H -> C critical-care escalation, distinct from
    ``omega`` which is the I_s -> H hospitalisation ratio. The 2025 chapter
    overloaded omega for both transitions.

    Note on units: ``alpha``, ``d_s``, ``d_a``, ``d_H`` are RATES with units
    of inverse-days. Their reciprocals are the corresponding clinical periods
    (e.g. 1/alpha = 5 days is the incubation period).
    """
    rho: float = 0.80          # symptomatic proportion (Boddington et al. 2021)
    alpha: float = 1.0 / 5.0   # incubation rate, day^-1 (Lauer et al. 2020)
    d_s: float = 1.0 / 4.0     # symptomatic infectious rate, day^-1 (Docherty 2020)
    d_a: float = 1.0 / 7.0     # asymptomatic infectious rate, day^-1 (Byrne 2020)
    d_H: float = 1.0 / 13.4    # hospitalisation outflow rate, day^-1 (Byrne 2020)
    mu: float = 0.05           # in-hospital mortality rate (calibrated, fallback)
    omega: float = 0.10        # I_s -> H hospitalisation ratio (calibrated)
    phi: float = 0.30          # H -> C critical-care escalation ratio (calibrated)


# ---------------------------------------------------------------------------
# Network architectures
# ---------------------------------------------------------------------------

class _MLP(nn.Module):
    """Fully connected MLP with tanh activations and sigmoid output.

    Matches the book chapter's specification: hidden width 20, tanh hidden,
    sigmoid output. Xavier initialisation.
    """

    def __init__(self, in_dim: int, hidden_layers: int, out_dim: int,
                 hidden_width: int = 20):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden_width), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers.extend([nn.Linear(hidden_width, hidden_width), nn.Tanh()])
        layers.append(nn.Linear(hidden_width, out_dim))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

        # Xavier initialisation for linear layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, t: Tensor) -> Tensor:  # noqa: D401
        return self.net(t)


class StateNet(_MLP):
    """t -> normalised (S, E, I_a, I_s, H, C, R, D) on [0,1]."""

    def __init__(self):
        super().__init__(in_dim=1, hidden_layers=5, out_dim=8)


class ParameterNet(_MLP):
    """t -> (beta, gamma_c, delta_c, eta) on [0,1]."""

    def __init__(self):
        super().__init__(in_dim=1, hidden_layers=3, out_dim=4)


class RegionalPINN(nn.Module):
    """Per-region PINN combining state and parameter networks."""

    def __init__(self, population: float, t_min: float = 0.0, t_max: float = 1.0):
        """
        Args:
            population: regional total population N_r, used for state denormalisation
            t_min, t_max: time normalisation bounds (network expects t in [t_min, t_max])
        """
        super().__init__()
        self.population = population
        self.t_min = t_min
        self.t_max = t_max
        self.state_net = StateNet()
        self.param_net = ParameterNet()

    def normalise_t(self, t: Tensor) -> Tensor:
        return (t - self.t_min) / (self.t_max - self.t_min + 1e-12)

    def forward(self, t: Tensor) -> tuple[Tensor, Tensor]:
        """Return (state_normalised, params_normalised) at time t.

        Both outputs are in [0,1] from sigmoid; downstream code is responsible
        for denormalising state by population and scaling params if needed.
        """
        t_norm = self.normalise_t(t).unsqueeze(-1) if t.ndim == 1 else self.normalise_t(t)
        state = self.state_net(t_norm)         # (..., 8)
        params = self.param_net(t_norm)        # (..., 4)
        return state, params

    def state_in_population(self, state_normalised: Tensor) -> Tensor:
        """Convert normalised compartments back to people-counts."""
        return state_normalised * self.population


# ---------------------------------------------------------------------------
# ODE residuals
# ---------------------------------------------------------------------------

def _seird_rhs(state: Tensor, params: Tensor, fixed: SEIRDFixedParams,
               population: float) -> Tensor:
    """Right-hand side of the SEIRD ODE system.

    Refinement: ``omega`` is the I_s -> H hospitalisation ratio only;
    a separate parameter ``phi`` handles H -> C critical-care escalation.
    Of those leaving hospital alive (rate d_H), proportion phi escalate to C
    and (1 - phi) recover. In-hospital mortality is governed by the separate
    rate mu.

    Args:
        state: (..., 8) tensor of (S, E, I_a, I_s, H, C, R, D) in [0,1]
        params: (..., 4) tensor of (beta, gamma_c, delta_c, eta) in [0,1]
        fixed: fixed parameters
        population: total population (for normalising the SI term)

    Returns:
        (..., 8) tensor of compartment derivatives.
    """
    S, E, I_a, I_s, H, C, R, D = state.unbind(dim=-1)
    beta, gamma_c, delta_c, eta = params.unbind(dim=-1)

    # SI term: beta * S * (I_s + I_a) / N. Note state is normalised by N already
    # so the (I_s + I_a) / N is just (I_s + I_a) in our [0,1] convention.
    SI = beta * S * (I_s + I_a)

    dS = -SI + eta * R
    dE = SI - fixed.alpha * E
    dI_s = fixed.alpha * fixed.rho * E - fixed.d_s * I_s
    dI_a = fixed.alpha * (1.0 - fixed.rho) * E - fixed.d_a * I_a

    # H: inflow from I_s (omega is hospitalisation ratio), outflow via d_H
    # (recovery / escalation) and mu (death).
    dH = fixed.d_s * fixed.omega * I_s - fixed.d_H * H - fixed.mu * H

    # C: inflow is phi * d_H * H (escalation from hospital), not (1 - omega).
    dC = fixed.phi * fixed.d_H * H - gamma_c * C - delta_c * C

    # R: receives non-hospitalised symptomatic, asymptomatic, non-escalating
    # hospital survivors (1 - phi) * d_H * H, and ICU survivors gamma_c * C.
    dR = (fixed.d_s * (1.0 - fixed.omega) * I_s
          + fixed.d_a * I_a
          + (1.0 - fixed.phi) * fixed.d_H * H
          + gamma_c * C
          - eta * R)

    # D: hospital deaths plus ICU deaths.
    dD = fixed.mu * H + delta_c * C

    return torch.stack([dS, dE, dI_a, dI_s, dH, dC, dR, dD], dim=-1)


def _autograd_state_derivative(pinn: RegionalPINN, t: Tensor) -> Tensor:
    """Compute dU/dt via PyTorch autograd at points t.

    Returns shape (..., 8).
    """
    t = t.requires_grad_(True)
    state, _ = pinn(t)
    # Sum each compartment across batch then differentiate; produces a Jacobian-free
    # gradient that we accumulate into an (N, 8) tensor.
    state_sum = state.sum(dim=0)  # (8,)
    grads = []
    for k in range(8):
        g = torch.autograd.grad(
            state_sum[k], t, retain_graph=True, create_graph=True,
            allow_unused=False,
        )[0]
        grads.append(g)
    # grads is a list of 8 tensors of shape (..., 1) — stack along last dim
    return torch.stack([g.squeeze(-1) for g in grads], dim=-1)


# ---------------------------------------------------------------------------
# PINN loss
# ---------------------------------------------------------------------------

# Indices into the 8-compartment state vector
COMPARTMENT_IDX = {"S": 0, "E": 1, "I_a": 2, "I_s": 3, "H": 4, "C": 5, "R": 6, "D": 7}
OBSERVABLE_COMPARTMENTS = ("I_s", "H", "C", "D")  # data fit only on these


def pinn_loss(
    pinn: RegionalPINN,
    t_data: Tensor,
    y_data: dict[str, Tensor],
    t_collocation: Tensor,
    fixed: SEIRDFixedParams,
    lambda_ode: float = 0.1,
) -> tuple[Tensor, dict[str, float]]:
    """Composite PINN loss for one region.

    Args:
        pinn: RegionalPINN instance
        t_data: (N_data,) tensor of observation times
        y_data: dict mapping compartment names ('I_s', 'H', 'C', 'D') to
                (N_data,) tensors of observed values, normalised by population.
        t_collocation: (N_coll,) tensor of collocation points (random in [t_min, t_max])
        fixed: fixed compartment parameters
        lambda_ode: weight on ODE residual

    Returns:
        Total loss (scalar), dict of component losses for logging.
    """
    # Data fit
    state_at_data, _ = pinn(t_data)
    data_loss = torch.tensor(0.0, device=t_data.device)
    for name in OBSERVABLE_COMPARTMENTS:
        if name in y_data:
            idx = COMPARTMENT_IDX[name]
            pred = state_at_data[..., idx]
            target = y_data[name]
            data_loss = data_loss + torch.mean((pred - target) ** 2)

    # ODE residual at collocation points
    state_coll, params_coll = pinn(t_collocation)
    dudt_pred = _autograd_state_derivative(pinn, t_collocation)
    rhs_pred = _seird_rhs(state_coll, params_coll, fixed, pinn.population)
    ode_residual = dudt_pred - rhs_pred
    ode_loss = torch.mean(ode_residual ** 2)

    total = data_loss + lambda_ode * ode_loss
    parts = {
        "data": float(data_loss.detach()),
        "ode": float(ode_loss.detach()),
        "total": float(total.detach()),
    }
    return total, parts


# ---------------------------------------------------------------------------
# Smoke test (run as script)
# ---------------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover
    torch.manual_seed(0)

    pinn = RegionalPINN(population=9_002_488.0)  # London-ish
    t = torch.linspace(0, 1, 50)
    state, params = pinn(t)
    print(f"State shape: {state.shape}; params shape: {params.shape}")
    print(f"State mean: {state.mean().item():.4f}; params mean: {params.mean().item():.4f}")

    # Synthetic 'data' for a few observable compartments
    fake_data = {
        "I_s": torch.full((50,), 0.001),
        "H":   torch.full((50,), 0.0001),
        "C":   torch.full((50,), 0.00002),
        "D":   torch.full((50,), 0.00001),
    }
    t_coll = torch.rand(100)
    fixed = SEIRDFixedParams()

    loss, parts = pinn_loss(pinn, t, fake_data, t_coll, fixed)
    print(f"Smoke test loss: {parts}")
