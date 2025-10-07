from __future__ import annotations

import argparse
import os
import re
import sys
import time
import webbrowser
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

EXAMPLES_DIR = HERE / "examples"

try:  # GUI is optional; CLI should still run without PySide6 installed.
    from PySide6.QtCore import QProcess, Qt, QUrl
    from PySide6.QtGui import QDesktopServices
    from PySide6.QtWidgets import (
        QApplication,
        QComboBox,
        QHBoxLayout,
        QLabel,
        QMessageBox,
        QPushButton,
        QPlainTextEdit,
        QVBoxLayout,
        QWidget,
    )
except ImportError:
    GUI_AVAILABLE = False
else:
    GUI_AVAILABLE = True

from alone_python_dial_mpc_core_algm import DIALConfig, DIALMPC
from alone_python_dial_mpc_env_base import BraxSimulator, SimulatorConfig
from alone_python_dial_mpc_env_biped import BipedSimulatorConfig
from alone_python_dial_mpc_env_quadruped import QuadrupedSimulatorConfig  # noqa: F401
from alone_python_dial_mpc_utils import get_example_path

try:
    from examples import examples as EXAMPLE_REGISTRY
except Exception:  # pragma: no cover - fallback to legacy package layout
    try:
        from dial_mpc.examples import examples as EXAMPLE_REGISTRY
    except Exception:  # pragma: no cover - only triggers if embedded assets missing
        EXAMPLE_REGISTRY: List[str] = []
HTML_PATH_PATTERN = re.compile(r"Visualization HTML saved to (.+)")
HTML_URL_PATTERN = re.compile(r"Visualization HTML URL: (.+)")
GIF_PATH_PATTERN = re.compile(r"Visualization GIF saved to (.+)")


@dataclass
class ExperimentConfig:
    n_steps: int = 100
    seed: int = 0
    output_dir: str = "output"
    dial: DIALConfig = field(default_factory=DIALConfig)
    sim: SimulatorConfig = field(default_factory=BipedSimulatorConfig)
    log_frequency: int = 10
    render_html: bool = True
    open_browser: bool = True


def discover_examples() -> List[str]:
    discovered: Dict[str, None] = {name: None for name in EXAMPLE_REGISTRY}
    if EXAMPLES_DIR.exists():
        for yaml_path in EXAMPLES_DIR.glob("*.yaml"):
            discovered.setdefault(yaml_path.stem, None)
    return sorted(discovered.keys())


def load_yaml_config(path: Optional[str | Path]) -> Dict[str, Any]:
    if path is None:
        return {}
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def merge_config(default: ExperimentConfig, override: Dict[str, Any]) -> ExperimentConfig:
    if not override:
        return default
    override = dict(override)

    dial_fields = set(DIALConfig.__dataclass_fields__.keys())
    sim_fields = set(SimulatorConfig.__dataclass_fields__.keys())
    top_level_fields = set(default.__dataclass_fields__.keys())

    dial_cfg = dict(override.pop("dial", {}) or {})
    sim_cfg = dict(override.pop("sim", {}) or {})
    top_level_updates: Dict[str, Any] = {}

    for key in list(override.keys()):
        if key in {"dial", "sim"}:
            continue
        value = override.pop(key)
        if key in top_level_fields:
            top_level_updates[key] = value
        elif key in dial_fields:
            dial_cfg.setdefault(key, value)
        elif key in sim_fields:
            sim_cfg.setdefault(key, value)
        else:
            sim_cfg.setdefault("extra_config", {}).setdefault(key, value)

    extra_config = sim_cfg.setdefault("extra_config", {})
    for key, value in override.items():
        extra_config.setdefault(key, value)

    dial = _replace_dataclass(default.dial, dial_cfg)
    sim = _replace_dataclass(default.sim, sim_cfg)
    return replace(default, dial=dial, sim=sim, **top_level_updates)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DIAL-MPC experiment.")
    parser.add_argument("--config", type=str, help="YAML configuration file path.")
    parser.add_argument("--example", type=str, help="Built-in example name.")
    parser.add_argument("--steps", type=int, help="Override number of MPC steps.")
    parser.add_argument("--seed", type=int, help="Random seed override.")
    parser.add_argument("--env", type=str, help="Override simulator environment name.")
    parser.add_argument(
        "--log-frequency",
        type=int,
        default=None,
        help="How often to print rollout progress (in steps).",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Launch PySide6 GUI instead of running the CLI workflow.",
    )
    parser.add_argument(
        "--no-save", action="store_true", help="Disable saving rollout artifacts."
    )
    parser.add_argument(
        "--render-html",
        dest="render_html",
        action="store_true",
        help="Force-enable HTML visualization export.",
    )
    parser.add_argument(
        "--no-render-html",
        dest="render_html",
        action="store_false",
        help="Disable HTML visualization export.",
    )
    parser.add_argument(
        "--open-browser",
        dest="open_browser",
        action="store_true",
        help="Open the rendered HTML in the default browser.",
    )
    parser.add_argument(
        "--no-open-browser",
        dest="open_browser",
        action="store_false",
        help="Prevent automatic browser launch after rendering.",
    )
    parser.add_argument(
        "--run-core",
        action="store_true",
        help=argparse.SUPPRESS,  # internal flag used by the GUI subprocess
    )
    parser.set_defaults(render_html=None, open_browser=None)
    return parser.parse_args(argv)


def _replace_dataclass(instance, values: Dict[str, Any]):
    if not values:
        return instance
    field_names = instance.__dataclass_fields__.keys()
    filtered = {k: v for k, v in values.items() if k in field_names}
    return replace(instance, **filtered)


def _resolve_config_sources(args: argparse.Namespace) -> Dict[str, Any]:
    config_data: Dict[str, Any] = {}
    if args.example:
        example_path = get_example_path(args.example)
        config_data.update(load_yaml_config(example_path))
    if args.config:
        config_data.update(load_yaml_config(args.config))
    return config_data


def run_experiment(cfg: ExperimentConfig, *, save: bool = True) -> Dict[str, Any]:
    rng = np.random.default_rng(cfg.seed)
    simulator = BraxSimulator(cfg.sim)
    dial = DIALMPC(cfg.dial, simulator)

    nodes = dial.initial_nodes()
    state = simulator.reset(cfg.seed)

    brax_states = []
    initial_brax_state = state.info.get("_brax_state")
    if initial_brax_state is not None:
        brax_states.append(initial_brax_state)

    rewards: List[float] = []
    controls: List[np.ndarray] = []
    node_history: List[np.ndarray] = []

    log_every = max(1, cfg.log_frequency) if cfg.log_frequency else 0
    print(
        f"Starting MPC rollout for {cfg.n_steps} steps; logging every "
        f"{log_every if log_every else 'N/A'} step(s)."
    )
    for step in range(cfg.n_steps):
        if step == 0:
            print("Priming JAX compilation on first step; this may take a moment...")
            sys.stdout.flush()
        control = dial.first_control(nodes)
        controls.append(control)
        state = simulator.step(state, control)
        brax_state = state.info.get("_brax_state")
        if brax_state is not None:
            brax_states.append(brax_state)
        rewards.append(state.reward)
        nodes = dial.shift_nodes(nodes)
        n_diffuse = cfg.dial.Ndiffuse_init if step == 0 else cfg.dial.Ndiffuse
        nodes, _ = dial.plan(state, nodes, n_diffuse, rng)
        node_history.append(nodes.copy())
        if log_every and ((step + 1) % log_every == 0 or step + 1 == cfg.n_steps):
            print(
                f"[step {step + 1:04d}/{cfg.n_steps:04d}] "
                f"reward={state.reward:.2e}"
            )
            sys.stdout.flush()

    mean_reward = float(np.mean(rewards)) if rewards else 0.0
    print(f"mean reward = {mean_reward:.2e}")

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir = output_dir.resolve()
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    if save:
        save_rollout(
            cfg,
            simulator.action_dim,
            rewards,
            controls,
            node_history,
            brax_states,
            output_dir,
            timestamp,
        )

    html_path: Optional[Path] = None
    gif_path: Optional[Path] = None
    if cfg.render_html and brax_states:
        html_path, gif_path = generate_visualization(
            simulator,
            brax_states,
            output_dir,
            timestamp,
            open_browser=cfg.open_browser,
        )

    return {
        "output_dir": output_dir,
        "timestamp": timestamp,
        "html_path": html_path,
        "gif_path": gif_path,
    }


def save_rollout(
    cfg: ExperimentConfig,
    action_dim: int,
    rewards: List[float],
    controls: List[np.ndarray],
    nodes: List[np.ndarray],
    brax_states,
    output_dir: Path,
    timestamp: str,
) -> None:
    np.save(output_dir / f"{timestamp}_rewards.npy", np.array(rewards))
    if controls:
        np.save(output_dir / f"{timestamp}_controls.npy", np.stack(controls, axis=0))
    if nodes:
        np.save(output_dir / f"{timestamp}_nodes.npy", np.stack(nodes, axis=0))

    if brax_states:
        pipeline_history = [
            bs.pipeline_state for bs in brax_states if hasattr(bs, "pipeline_state")
        ]
    else:
        pipeline_history = []

    if pipeline_history:
        qpos = np.stack(
            [np.asarray(getattr(ps, "qpos", ps.q)) for ps in pipeline_history], axis=0
        )
        qvel = np.stack(
            [np.asarray(getattr(ps, "qvel", ps.qd)) for ps in pipeline_history], axis=0
        )
        ctrl_arrays = []
        for ps in pipeline_history:
            ctrl = getattr(ps, "ctrl", None)
            if ctrl is None or np.asarray(ctrl).size == 0:
                ctrl_arrays.append(np.zeros(action_dim))
            else:
                ctrl_arrays.append(np.asarray(ctrl))
        ctrl = np.stack(ctrl_arrays, axis=0)
        np.savez(
            output_dir / f"{timestamp}_pipeline_states.npz",
            qpos=qpos,
            qvel=qvel,
            ctrl=ctrl,
        )

    with open(output_dir / f"{timestamp}_config.yaml", "w", encoding="utf-8") as fh:
        yaml.safe_dump(
            {
                "n_steps": cfg.n_steps,
                "seed": cfg.seed,
                "output_dir": cfg.output_dir,
                "log_frequency": cfg.log_frequency,
                "render_html": cfg.render_html,
                "open_browser": cfg.open_browser,
                "dial": cfg.dial.__dict__,
                "sim": cfg.sim.__dict__,
            },
            fh,
        )


def generate_visualization(
    simulator: BraxSimulator,
    brax_states,
    output_dir: Path,
    timestamp: str,
    *,
    open_browser: bool,
) -> Tuple[Optional[Path], Optional[Path]]:
    pipeline_states = [
        bs.pipeline_state for bs in brax_states if hasattr(bs, "pipeline_state")
    ]
    html_path: Optional[Path] = None
    gif_path: Optional[Path] = None

    if pipeline_states:
        states_for_render = (
            pipeline_states[1:] if len(pipeline_states) > 1 else pipeline_states
        )
    else:
        states_for_render = []

    if states_for_render:
        html_str = simulator.render_html(states_for_render, height=720, colab=True)
        html_path = output_dir / f"{timestamp}_brax_visualization.html"
        with html_path.open("w", encoding="utf-8") as fh:
            fh.write(html_str)
        print(f"Visualization HTML saved to {html_path}")
        print(f"Visualization HTML URL: {html_path.as_uri()}")
        print(
            "Note: HTML viewer loads JavaScript from public CDNs; ensure the viewing "
            "machine has internet access or host equivalent assets locally."
        )
        if open_browser:
            webbrowser.open(html_path.as_uri())

    if brax_states:
        try:
            gif_bytes = simulator.render_gif(brax_states, height=360, width=640)
        except Exception as exc:  # pragma: no cover - visualization fallback
            print(f"Failed to generate visualization GIF: {exc}")
        else:
            gif_path = output_dir / f"{timestamp}_brax_animation.gif"
            gif_path.write_bytes(gif_bytes)
            print(f"Visualization GIF saved to {gif_path}")

    return html_path, gif_path
    html_path = output_dir / f"{timestamp}_brax_visualization.html"
    with html_path.open("w", encoding="utf-8") as fh:
        fh.write(html_str)
    print(f"Visualization HTML saved to {html_path}")
    print(f"Visualization HTML URL: {html_path.as_uri()}")
    if open_browser:
        webbrowser.open(html_path.as_uri())
    return html_path


if GUI_AVAILABLE:

    class DialMpcWindow(QWidget):
        """Simple GUI to launch examples and monitor CLI output."""

        def __init__(self) -> None:
            super().__init__()
            self.setWindowTitle("DIAL-MPC Example Launcher")
            self.process: QProcess | None = None
            self.visual_uri: Optional[str] = None
            self._build_ui()
            self._load_examples()

        def _build_ui(self) -> None:
            layout = QVBoxLayout()

            heading = QLabel("Select an example to run")
            heading.setAlignment(Qt.AlignCenter)
            heading.setStyleSheet("font-size: 18px; font-weight: bold;")
            layout.addWidget(heading)

            chooser_layout = QHBoxLayout()
            chooser_label = QLabel("Example name or YAML:")
            chooser_layout.addWidget(chooser_label)

            self.example_combo = QComboBox()
            self.example_combo.setEditable(True)
            chooser_layout.addWidget(self.example_combo, stretch=1)

            self.refresh_button = QPushButton("Refresh")
            self.refresh_button.clicked.connect(self._load_examples)
            chooser_layout.addWidget(self.refresh_button)

            layout.addLayout(chooser_layout)

            buttons_layout = QHBoxLayout()
            self.run_button = QPushButton("Run Example")
            self.run_button.clicked.connect(self._start_process)
            buttons_layout.addWidget(self.run_button)

            self.stop_button = QPushButton("Stop")
            self.stop_button.clicked.connect(self._stop_process)
            self.stop_button.setEnabled(False)
            buttons_layout.addWidget(self.stop_button)

            self.open_browser_button = QPushButton("Open Visualization")
            self.open_browser_button.clicked.connect(self._open_visualization)
            self.open_browser_button.setEnabled(False)
            buttons_layout.addWidget(self.open_browser_button)

            layout.addLayout(buttons_layout)

            log_label = QLabel("Process output:")
            layout.addWidget(log_label)

            self.log_view = QPlainTextEdit()
            self.log_view.setReadOnly(True)
            layout.addWidget(self.log_view, stretch=1)

            self.status_label = QLabel("Idle")
            self.status_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            layout.addWidget(self.status_label)

            self.setLayout(layout)

        def _load_examples(self) -> None:
            examples = discover_examples()
            current_text = self.example_combo.currentText()
            self.example_combo.blockSignals(True)
            self.example_combo.clear()
            self.example_combo.addItems(examples)
            self.example_combo.setCurrentText(current_text)
            self.example_combo.blockSignals(False)
            self._append_log(f"Loaded {len(examples)} examples from repository")

        def _append_log(self, message: str) -> None:
            self.log_view.appendPlainText(message)
            self.log_view.verticalScrollBar().setValue(
                self.log_view.verticalScrollBar().maximum()
            )

        def _start_process(self) -> None:
            if self.process and self.process.state() != QProcess.NotRunning:
                QMessageBox.warning(
                    self,
                    "Process running",
                    "An example is already running. Stop it before launching another.",
                )
                return

            chosen = self.example_combo.currentText().strip()
            if not chosen:
                QMessageBox.warning(
                    self, "Missing input", "Please select or enter an example."
                )
                return

            args: List[str]
            if chosen.endswith(".yaml"):
                config_path = Path(chosen)
                if not config_path.is_absolute():
                    config_path = EXAMPLES_DIR / config_path
                if not config_path.exists():
                    QMessageBox.critical(
                        self,
                        "Config not found",
                        f"Could not locate config file: {config_path}",
                    )
                    return
                args = ["--config", str(config_path.resolve())]
                self._append_log(f"Launching DIAL-MPC with config {config_path}")
            else:
                args = ["--example", chosen]
                self._append_log(f"Launching DIAL-MPC example '{chosen}'")

            self.process = QProcess(self)
            env = self.process.processEnvironment()
            pythonpath = env.value("PYTHONPATH", "")
            repo_path = str(HERE.resolve())
            if pythonpath:
                env.insert("PYTHONPATH", os.pathsep.join([repo_path, pythonpath]))
            else:
                env.insert("PYTHONPATH", repo_path)
            self.process.setProcessEnvironment(env)

            self.process.setProgram(sys.executable)
            script_path = str(Path(__file__).resolve())
            self.process.setArguments(
                [script_path, "--run-core", "--no-open-browser", *args]
            )

            self.process.setProcessChannelMode(QProcess.MergedChannels)
            self.process.readyReadStandardOutput.connect(self._handle_stdout)
            self.process.finished.connect(self._process_finished)
            self.process.errorOccurred.connect(self._process_error)

            self.visual_uri = None
            self.open_browser_button.setEnabled(False)

            self.process.start()

            if not self.process.waitForStarted(5000):
                error_message = self.process.errorString()
                self._append_log(f"Failed to start process: {error_message}")
                QMessageBox.critical(self, "Launch failed", error_message)
                self.process = None
                return

            self.run_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.status_label.setText("Running")

        def _handle_stdout(self) -> None:
            if not self.process:
                return
            output = self.process.readAllStandardOutput().data().decode(
                "utf-8", "ignore"
            )
            if output:
                self._append_log(output.rstrip())
                self._check_for_visualization_url(output)

        def _stop_process(self) -> None:
            if not self.process:
                return
            if self.process.state() == QProcess.NotRunning:
                return

            self._append_log("Terminating process...")
            self.process.terminate()
            if not self.process.waitForFinished(3000):
                self._append_log("Process did not terminate gracefully; killing.")
                self.process.kill()
                self.process.waitForFinished()

        def _process_finished(self, exit_code: int, exit_status: QProcess.ExitStatus) -> None:
            if exit_status == QProcess.NormalExit:
                self._append_log(f"Process finished with exit code {exit_code}")
            else:
                self._append_log("Process crashed")
            self.run_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.open_browser_button.setEnabled(self.visual_uri is not None)
            self.status_label.setText("Idle")
            self.process = None

        def _process_error(self, error: QProcess.ProcessError) -> None:
            self._append_log(f"Process error: {error}")
            self.run_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.open_browser_button.setEnabled(self.visual_uri is not None)
            self.status_label.setText("Idle")
            self.process = None

        def _check_for_visualization_url(self, output: str) -> None:
            if self.visual_uri is not None:
                return
            for line in output.splitlines():
                path_match = HTML_PATH_PATTERN.search(line)
                url_match = HTML_URL_PATTERN.search(line)
                gif_match = GIF_PATH_PATTERN.search(line)
                if path_match:
                    path = Path(path_match.group(1).strip())
                    if path.exists():
                        self.visual_uri = path.as_uri()
                        break
                elif url_match:
                    self.visual_uri = url_match.group(1).strip()
                    break
                elif gif_match:
                    path = Path(gif_match.group(1).strip())
                    if path.exists() and self.visual_uri is None:
                        self.visual_uri = path.as_uri()
                        break
            if self.visual_uri:
                self.open_browser_button.setEnabled(True)
                self._append_log(f"Visualization ready: {self.visual_uri}")
                self._open_visualization(auto=True)

        def _open_visualization(self, auto: bool = False) -> None:
            if not self.visual_uri:
                if not auto:
                    QMessageBox.information(
                        self,
                        "Visualization not ready",
                        "No visualization output detected yet.",
                    )
                return
            opened = QDesktopServices.openUrl(QUrl(self.visual_uri))
            if opened and not auto:
                self._append_log(f"Opened browser at {self.visual_uri}")
            elif not opened:
                self._append_log(
                    f"Could not open default browser. Please visit {self.visual_uri} manually."
                )

        def closeEvent(self, event) -> None:  # noqa: D401
            if self.process and self.process.state() != QProcess.NotRunning:
                self._stop_process()
            super().closeEvent(event)


def launch_gui() -> None:
    if not GUI_AVAILABLE:
        raise RuntimeError("PySide6 is not installed; GUI mode is unavailable.")
    app = QApplication(sys.argv)
    window = DialMpcWindow()
    window.resize(800, 600)
    window.show()
    sys.exit(app.exec())


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    if not args.run_core:
        if args.gui:
            launch_gui()
            return
        if argv is None and len(sys.argv) == 1:
            # No CLI arguments provided: default to GUI for an interactive experience.
            launch_gui()
            return

    default_cfg = ExperimentConfig()
    config_overrides = _resolve_config_sources(args)
    cfg = merge_config(default_cfg, config_overrides)

    if args.steps is not None:
        cfg = replace(cfg, n_steps=args.steps)
    if args.seed is not None:
        cfg = replace(cfg, seed=args.seed, dial=replace(cfg.dial, seed=args.seed))
    if args.env is not None:
        cfg = replace(cfg, sim=replace(cfg.sim, env_name=args.env))
    if args.log_frequency is not None:
        cfg = replace(cfg, log_frequency=args.log_frequency)
    if args.render_html is not None:
        cfg = replace(cfg, render_html=args.render_html)
    if args.open_browser is not None:
        cfg = replace(cfg, open_browser=args.open_browser)

    result = run_experiment(cfg, save=not args.no_save)
    html_path = result.get("html_path")
    if html_path:
        print(f"Visualization generated at: {html_path}")
    gif_path = result.get("gif_path")
    if gif_path:
        print(f"Animation GIF available at: {gif_path}")


if __name__ == "__main__":
    main()
