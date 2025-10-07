from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

import jax
import jax.numpy as jnp
from brax import math


# ---------------------------------------------------------------------------
# Kinematics helpers
# ---------------------------------------------------------------------------

def global_to_body_velocity(v, q):
    """将全局速度转换到机体坐标系。"""
    return math.inv_rotate(v, q)


def body_to_global_velocity(v, q):
    """将机体速度转换回全局坐标系。"""
    return math.rotate(v, q)


@jax.jit
def get_foot_step(duty_ratio, cadence, amplitude, phases, time):
    """根据步态参数生成足端高度曲线。"""

    def step_height(t, footphase, duty_ratio):
        angle = (t + jnp.pi - footphase) % (2 * jnp.pi) - jnp.pi
        angle = jnp.where(duty_ratio < 1, angle * 0.5 / (1 - duty_ratio), angle)
        clipped_angle = jnp.clip(angle, -jnp.pi / 2, jnp.pi / 2)
        value = jnp.where(duty_ratio < 1, jnp.cos(clipped_angle), 0)
        final_value = jnp.where(jnp.abs(value) >= 1e-6, jnp.abs(value), 0.0)
        return final_value

    h_steps = amplitude * jax.vmap(step_height, in_axes=(None, 0, None))(
        time * 2 * jnp.pi * cadence + jnp.pi,
        2 * jnp.pi * phases,
        duty_ratio,
    )
    return h_steps


# ---------------------------------------------------------------------------
# Resource discovery helpers
# ---------------------------------------------------------------------------

def _script_root() -> Path:
    main = sys.modules.get("__main__")
    if main and getattr(main, "__file__", None):
        return Path(main.__file__).resolve().parent
    return Path.cwd()


def _candidate_roots() -> list[Path]:
    root = _script_root()
    candidates = [root]
    candidates.extend(root.parents)
    extra = set()
    extra.add(root / "dial_mpc")
    extra.add(root / "dial_mpc-simple")
    extra.add(root / "dial_mpc-simple" / "dial_mpc")
    extra.add(root / "dial-mpc-simple")
    extra.add(root / "dial-mpc-simple" / "dial_mpc")
    for parent in root.parents:
        extra.add(parent / "dial_mpc")
        extra.add(parent / "dial_mpc-simple" / "dial_mpc")
        extra.add(parent / "dial-mpc-simple" / "dial_mpc")
        extra.add(parent / "dial_mpc-simple")
    candidates.extend(extra)
    return [c for c in candidates if c]


def get_model_path(robot_name: str, model_name: str) -> Path:
    """定位机器人模型文件。"""
    for base in _candidate_roots():
        for rel in [
            Path("models"),
            Path("dial_mpc") / "models",
            Path("robots"),
            Path("dial-mpc") / "models",
        ]:
            candidate = (base / rel / robot_name / model_name).resolve()
            if candidate.exists():
                return candidate
    raise FileNotFoundError(
        f"Cannot locate model '{model_name}' for robot '{robot_name}'."
    )


def get_example_path(example_name: str) -> Path:
    """定位示例配置文件。"""
    example_name = example_name if example_name.endswith(".yaml") else example_name + ".yaml"
    for base in _candidate_roots():
        for rel in [
            Path("examples"),
            Path("dial_mpc") / "examples",
            Path("dial-mpc") / "examples",
        ]:
            candidate = (base / rel / example_name).resolve()
            if candidate.exists():
                return candidate
    raise FileNotFoundError(
        f"Cannot locate example YAML '{example_name}'."
    )


def load_dataclass_from_dict(
    dataclass_type,
    data_dict: Dict[str, Any],
    convert_list_to_array: bool = False,
):
    """将字典数据加载为 dataclass，支持自动转换 JAX 数组。"""
    keys = dataclass_type.__dataclass_fields__.keys() & data_dict.keys()
    kwargs = {key: data_dict[key] for key in keys}
    if convert_list_to_array:
        for key, value in list(kwargs.items()):
            if isinstance(value, list):
                kwargs[key] = jnp.array(value)
    return dataclass_type(**kwargs)


LegID = {
    "FR_0": 0,
    "FR_1": 1,
    "FR_2": 2,
    "FL_0": 3,
    "FL_1": 4,
    "FL_2": 5,
    "RR_0": 6,
    "RR_1": 7,
    "RR_2": 8,
    "RL_0": 9,
    "RL_1": 10,
    "RL_2": 11,
}

HIGHLEVEL = 0xEE
LOWLEVEL = 0xFF
TRIGERLEVEL = 0xF0
PosStopF = 2.146e9
VelStopF = 16000.0
