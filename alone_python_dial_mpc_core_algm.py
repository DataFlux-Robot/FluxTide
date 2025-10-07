from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from scipy.interpolate import CubicSpline

if TYPE_CHECKING:  # pragma: no cover
    from alone_python_dial_mpc_env_base import RobotState, SimulatorInterface


@dataclass
class DIALConfig:
    seed: int = 0
    n_steps: int = 100
    Nsample: int = 2048
    Hsample: int = 16
    Hnode: int = 4
    Ndiffuse: int = 2
    Ndiffuse_init: int = 10
    temp_sample: float = 0.06
    horizon_diffuse_factor: float = 0.9
    traj_diffuse_factor: float = 0.5
    sigma_scale: float = 1.0
    ctrl_dt: float = 0.02
    action_clip: Tuple[float, float] = (-1.0, 1.0)
    update_method: str = "mppi"


class ControlNode:
    def __init__(self, config: DIALConfig, action_dim: int):
        self._cfg = config
        self._action_dim = action_dim
        horizon = config.ctrl_dt * config.Hsample
        self._t_nodes = np.linspace(0.0, horizon, config.Hnode + 1)
        self._t_controls = np.linspace(0.0, horizon, config.Hsample + 1)

    def nodes_to_control(self, nodes: np.ndarray) -> np.ndarray:
        spline = CubicSpline(self._t_nodes, nodes, axis=0, bc_type="natural")
        controls = spline(self._t_controls)
        lo, hi = self._cfg.action_clip
        return np.clip(controls, lo, hi)

    def controls_to_nodes(self, controls: np.ndarray) -> np.ndarray:
        spline = CubicSpline(self._t_controls, controls, axis=0, bc_type="natural")
        nodes = spline(self._t_nodes)
        lo, hi = self._cfg.action_clip
        return np.clip(nodes, lo, hi)

    def batch_nodes_to_control(self, nodes_batch: np.ndarray) -> np.ndarray:
        controls = [self.nodes_to_control(nodes) for nodes in nodes_batch]
        return np.stack(controls, axis=0)


class DIALMPC:
    def __init__(self, config: DIALConfig, simulator: "SimulatorInterface"):
        self._cfg = config
        self._sim = simulator
        self._action_dim = simulator.action_dim
        self._control_node = ControlNode(config, self._action_dim)
        base = np.power(
            self._cfg.horizon_diffuse_factor,
            np.arange(self._cfg.Hnode + 1)[::-1],
        )
        self._sigma_base = self._cfg.sigma_scale * base

    @property
    def action_dim(self) -> int:
        return self._action_dim

    def initial_nodes(self) -> np.ndarray:
        shape = (self._cfg.Hnode + 1, self._action_dim)
        return np.zeros(shape, dtype=np.float32)

    def noise_schedule(self, count: int) -> np.ndarray:
        if count <= 0:
            return np.empty((0, self._cfg.Hnode + 1))
        factors = np.power(self._cfg.traj_diffuse_factor, np.arange(count))
        return np.stack([self._sigma_base * f for f in factors], axis=0)

    def shift_nodes(self, nodes: np.ndarray) -> np.ndarray:
        controls = self._control_node.nodes_to_control(nodes)
        controls = np.roll(controls, -1, axis=0)
        controls[-1] = 0.0
        return self._control_node.controls_to_nodes(controls)

    def first_control(self, nodes: np.ndarray) -> np.ndarray:
        controls = self._control_node.nodes_to_control(nodes)
        return controls[0]

    def plan(
        self,
        state: "RobotState",
        nodes: np.ndarray,
        n_diffuse: int,
        rng: Optional[np.random.Generator] = None,
        *,
        collect_traces: bool = False,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        if rng is None:
            rng = np.random.default_rng()
        schedule = self.noise_schedule(n_diffuse)
        trace: List[Dict[str, Any]] = []
        current = nodes
        for sigma_vec in schedule:
            current, info = self._reverse_once(state, current, sigma_vec, rng)
            if collect_traces:
                trace.append(info)
        aux: Dict[str, Any] = {}
        if collect_traces:
            aux["diffusion_trace"] = trace
        return current, aux

    def _reverse_once(
        self,
        state: "RobotState",
        nodes: np.ndarray,
        sigma_vec: np.ndarray,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        samples = self._sample_nodes(nodes, sigma_vec, rng)
        controls = self._control_node.batch_nodes_to_control(samples)
        rewards, _ = self._sim.batch_rollout(state, controls, return_states=False)
        reward_mean = rewards.mean(axis=-1)
        baseline = reward_mean[-1]
        std = reward_mean.std()
        if std < 1e-8:
            weights = np.ones_like(reward_mean) / reward_mean.size
        else:
            scaled = (reward_mean - baseline) / (std * self._cfg.temp_sample)
            scaled = scaled - scaled.max()
            weights = np.exp(scaled)
            weights /= weights.sum()
        new_nodes = np.einsum("n,nij->ij", weights, samples)
        info = {
            "weights": weights,
            "reward_mean": reward_mean,
            "sigma": sigma_vec,
        }
        return new_nodes, info

    def _sample_nodes(
        self,
        nodes: np.ndarray,
        sigma_vec: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        base = np.repeat(nodes[None, ...], self._cfg.Nsample, axis=0)
        noise = rng.standard_normal(size=base.shape).astype(np.float32)
        sample = base + noise * sigma_vec[None, :, None]
        sample[:, 0, :] = nodes[0]
        lo, hi = self._cfg.action_clip
        sample = np.clip(sample, lo, hi)
        augmented = np.concatenate([sample, nodes[None]], axis=0)
        return augmented
