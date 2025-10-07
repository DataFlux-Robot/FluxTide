from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

try:
    import jax
    import jax.numpy as jnp
except ImportError as exc:  # pragma: no cover
    jax = None
    jnp = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

if _IMPORT_ERROR is None:
    try:
        from brax.base import System
        from brax.envs.base import PipelineEnv
    except ImportError as exc:  # pragma: no cover
        _IMPORT_ERROR = exc
        PipelineEnv = object  # type: ignore[assignment]
        System = Any  # type: ignore[assignment]
else:  # pragma: no cover
    PipelineEnv = object  # type: ignore[assignment]
    System = Any  # type: ignore[assignment]


ENV_MODULE_REGISTRY: Dict[str, Tuple[str, str]] = {}


def register_envs(mapping: Dict[str, Tuple[str, str]]) -> None:
    ENV_MODULE_REGISTRY.update(mapping)


@dataclass
class BaseEnvConfig:
    task_name: str = "default"
    randomize_tasks: bool = False
    kp: Union[float, jax.Array] = 30.0
    kd: Union[float, jax.Array] = 1.0
    debug: bool = False
    dt: float = 0.02
    timestep: float = 0.02
    backend: str = "mjx"
    leg_control: str = "torque"
    action_scale: float = 1.0


if _IMPORT_ERROR is None:

    class BaseEnv(PipelineEnv):
        def __init__(self, config: BaseEnvConfig):
            assert jnp.allclose(
                config.dt % config.timestep, 0.0
            ), "timestep must be divisible by dt"
            self._config = config
            n_frames = int(config.dt / config.timestep)
            sys = self.make_system(config)
            super().__init__(sys, config.backend, n_frames, config.debug)

            self.physical_joint_range = self.sys.jnt_range[1:]
            self.joint_range = self.physical_joint_range
            self.joint_torque_range = self.sys.actuator_ctrlrange

            self._nv = self.sys.nv
            self._nq = self.sys.nq

        def make_system(self, config: BaseEnvConfig) -> System:
            raise NotImplementedError

        @partial(jax.jit, static_argnums=(0,))
        def act2joint(self, act: jax.Array) -> jax.Array:
            act_normalized = (act * self._config.action_scale + 1.0) / 2.0
            joint_targets = self.joint_range[:, 0] + act_normalized * (
                self.joint_range[:, 1] - self.joint_range[:, 0]
            )
            joint_targets = jnp.clip(
                joint_targets,
                self.physical_joint_range[:, 0],
                self.physical_joint_range[:, 1],
            )
            return joint_targets

        @partial(jax.jit, static_argnums=(0,))
        def act2tau(self, act: jax.Array, pipeline_state) -> jax.Array:
            joint_target = self.act2joint(act)

            q = pipeline_state.qpos[7:]
            q = q[: len(joint_target)]
            qd = pipeline_state.qvel[6:]
            qd = qd[: len(joint_target)]
            q_err = joint_target - q
            tau = self._config.kp * q_err - self._config.kd * qd

            tau = jnp.clip(
                tau, self.joint_torque_range[:, 0], self.joint_torque_range[:, 1]
            )
            return tau

else:  # pragma: no cover

    class BaseEnv:  # type: ignore[override]
        def __init__(self, *_: Any, **__: Any):
            raise ImportError(
                "BaseEnv requires brax and jax to be installed"
            ) from _IMPORT_ERROR


@dataclass
class SimulatorConfig:
    env_name: str = "unitree_h1_walk"
    physics_dt: float = 0.005
    config_module: Optional[str] = None
    config_class: Optional[str] = None
    extra_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RobotState:
    qpos: np.ndarray
    qvel: np.ndarray
    reward: float
    info: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.info is None:
            self.info = {}


class SimulatorInterface(ABC):
    action_dim: int

    @abstractmethod
    def reset(self, seed: int) -> RobotState:
        raise NotImplementedError

    @abstractmethod
    def step(self, state: RobotState, action: np.ndarray) -> RobotState:
        raise NotImplementedError

    @abstractmethod
    def batch_rollout(
        self,
        initial_state: RobotState,
        action_sequences: np.ndarray,
        return_states: bool = False,
        parallel_backend: str = "auto",
        **kwargs: Any,
    ) -> Tuple[np.ndarray, Optional[List[List[RobotState]]]]:
        raise NotImplementedError


class BraxSimulator(SimulatorInterface):
    def __init__(self, config: SimulatorConfig, *, jit_compile: bool = True):
        if _IMPORT_ERROR is not None:
            raise ImportError(
                "BraxSimulator requires jax to be installed"
            ) from _IMPORT_ERROR

        from brax import envs as brax_envs

        env_config = self._resolve_env_config(config)
        self._env = brax_envs.get_environment(config.env_name, config=env_config)
        self._jit_reset = jax.jit(self._env.reset) if jit_compile else self._env.reset
        self._jit_step = jax.jit(self._env.step) if jit_compile else self._env.step
        self._jit_batch_rollout = self._compile_batch_rollout(jit_compile)
        self.action_dim = int(self._env.action_size)

    def reset(self, seed: int) -> RobotState:
        key = jax.random.PRNGKey(seed)
        state = self._jit_reset(key)
        return self._brax_to_robot_state(state)

    def step(self, state: RobotState, action: np.ndarray) -> RobotState:
        brax_state = state.info["_brax_state"]
        next_state = self._jit_step(brax_state, jnp.array(action))
        return self._brax_to_robot_state(next_state)

    def batch_rollout(
        self,
        initial_state: RobotState,
        action_sequences: np.ndarray,
        return_states: bool = False,
        parallel_backend: str = "auto",
        **kwargs: Any,
    ) -> Tuple[np.ndarray, Optional[List[List[RobotState]]]]:
        del parallel_backend, kwargs
        brax_state = initial_state.info["_brax_state"]
        actions_jax = jnp.array(action_sequences)
        rewards_jax, states_jax = self._jit_batch_rollout(brax_state, actions_jax)
        rewards = np.array(rewards_jax)
        if not return_states:
            return rewards, None
        trajectories = self._convert_batch_states(states_jax)
        return rewards, trajectories

    def render_html(
        self,
        pipeline_states,
        *,
        height: int = 720,
        colab: bool = True,
        base_url: str | None = None,
    ) -> str:
        from brax.io import html

        sys_config = self._env.sys.tree_replace({"opt.timestep": self._env.dt})
        return html.render(
            sys_config,
            list(pipeline_states),
            height=height,
            colab=colab,
            base_url=base_url,
        )

    def render_gif(
        self,
        brax_states,
        *,
        height: int = 360,
        width: int = 640,
        camera: str | None = None,
    ) -> bytes:
        import os
        from brax.io import image

        if not brax_states:
            raise ValueError("No Brax states provided for rendering.")
        backend = os.environ.get("MUJOCO_GL")
        if backend not in {"egl", "osmesa"}:
            raise RuntimeError(
                "MUJOCO_GL must be set to 'egl' or 'osmesa' for GIF rendering."
            )
        return image.render(
            self._env.sys,
            list(brax_states),
            height=height,
            width=width,
            camera=camera,
            fmt="gif",
        )

    def _resolve_env_config(self, config: SimulatorConfig) -> Any:
        module_name = config.config_module
        class_name = config.config_class
        if module_name is None and config.env_name in ENV_MODULE_REGISTRY:
            module_name, class_name = ENV_MODULE_REGISTRY[config.env_name]

        env_config = config.extra_config or {}
        if module_name is not None:
            module = importlib.import_module(module_name)
            self._maybe_register_environment(module)
            if class_name is not None:
                config_cls = getattr(module, class_name)
                kwargs = dict(env_config)
                if jnp is not None:
                    for key, value in list(kwargs.items()):
                        if isinstance(value, list):
                            kwargs[key] = jnp.array(value)
                return config_cls(**kwargs)
        return env_config or None

    def _maybe_register_environment(self, module: Any) -> None:
        _ = module

    def _compile_batch_rollout(self, jit_compile: bool):
        def rollout_single(brax_state, actions):
            def body_fn(carry, action):
                next_state = self._env.step(carry, action)
                return next_state, (next_state.reward, next_state)

            _, (rewards, states) = jax.lax.scan(body_fn, brax_state, actions)
            return rewards, states

        vmapped = jax.vmap(rollout_single, in_axes=(None, 0))
        if jit_compile:
            return jax.jit(vmapped)
        return vmapped

    def _brax_to_robot_state(self, brax_state) -> RobotState:
        pipeline_state = getattr(brax_state, "pipeline_state", None)
        if pipeline_state is None:
            raise AttributeError("Brax state does not contain pipeline_state.")
        qpos = np.array(getattr(pipeline_state, "qpos", pipeline_state.q))
        qvel = np.array(getattr(pipeline_state, "qvel", pipeline_state.qd))
        reward = float(np.array(brax_state.reward))
        info = {"_brax_state": brax_state, "_pipeline_state": pipeline_state}
        return RobotState(qpos=qpos, qvel=qvel, reward=reward, info=info)

    def _convert_batch_states(self, states_jax) -> List[List[RobotState]]:
        batch_size = states_jax.reward.shape[0]
        horizon = states_jax.reward.shape[1]
        trajectories: List[List[RobotState]] = []
        for batch_idx in range(batch_size):
            traj_states: List[RobotState] = []
            for step_idx in range(horizon):
                slice_state = jax.tree_map(
                    lambda x, b=batch_idx, s=step_idx: x[b, s],
                    states_jax,
                )
                traj_states.append(self._brax_to_robot_state(slice_state))
            trajectories.append(traj_states)
        return trajectories
