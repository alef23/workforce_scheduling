from __future__ import annotations

import argparse
import fcntl
import json
import os
import random
import re
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.dataset_generation import create_dataset_buffer_layout
from modules.evaluators.centralized import CentralizedEvaluatorConfig
from modules.evaluators.resnet.resnet_state_evaluator import ResNetStateEvaluator
from modules.learning import ResNetLearnerConfig, ResNetSampleLearner
from modules.mcts.mcts_schemas import MCTSConfig, MCTSMode
from modules.mcts_generation import (
    MCTSGenerationConfig,
    MCTSGenerationOrchestrator,
    MCTSOrchestratorConfig,
    MCTSStartMode,
    ReweightedPolicyConfig,
    build_mcts_generation_jobs,
)
from modules.storage import TrajectoryBuffer
from modules.workforce_engine.schemas import ProblemSetup


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Genera samples MCTS/reweighted desde el buffer stock_adjusted."
    )
    parser.add_argument(
        "--output-root",
        default="datasets",
        help="Directorio raiz de buffers. Default: datasets.",
    )
    parser.add_argument(
        "--source-path",
        default=None,
        help="Buffer stock_adjusted fuente. Default: <output-root>/derived/stock_adjusted/trajectories.zarr.",
    )
    parser.add_argument(
        "--sample-path",
        default=None,
        help="SampleBuffer destino. Default: <output-root>/samples/samples.zarr.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Cantidad de workers MCTS. Default: 1.",
    )
    parser.add_argument(
        "--n-trajectories",
        type=int,
        default=None,
        help="Cantidad opcional de trayectorias stock a procesar. Default: todas.",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Mezcla IDs fuente antes de aplicar --n-trajectories.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed opcional para seleccion de jobs y workers. Default: None.",
    )
    parser.add_argument(
        "--overwrite-samples",
        action="store_true",
        help="Recrea el SampleBuffer destino si ya existe.",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=1,
        help="Frecuencia de impresion de progreso. Default: 1 job.",
    )
    parser.add_argument(
        "--p-mcts",
        type=float,
        default=0.2,
        help="Probabilidad de generar samples con MCTS por trayectoria stock. Default: 0.2.",
    )
    parser.add_argument(
        "--start-mode",
        choices=[mode.value for mode in MCTSStartMode],
        default=MCTSStartMode.INITIAL_ONLY.value,
        help="Modo de seleccion de estados semilla MCTS.",
    )
    parser.add_argument(
        "--max-seed-states",
        type=int,
        default=0,
        help="Cantidad maxima de estados semilla adicionales. Default: 0.",
    )
    parser.add_argument(
        "--seed-state-probability",
        type=float,
        default=0.0,
        help="Probabilidad de seleccionar cada estado candidato. Default: 0.0.",
    )
    parser.add_argument(
        "--tail-window-size",
        type=int,
        default=None,
        help=(
            "Cantidad de estados anteriores al terminal considerados por "
            "tail_forward_sampled. Requerido para ese modo."
        ),
    )
    parser.add_argument(
        "--mcts-simulations",
        type=int,
        default=16,
        help="Simulaciones MCTS por decision. Default: 16.",
    )
    parser.add_argument(
        "--c-puct",
        type=float,
        default=1.0,
        help="Constante de exploracion PUCT. Default: 1.0.",
    )
    parser.add_argument(
        "--mcts-policy-weight",
        type=float,
        default=1.0,
        help="Policy weight para samples MCTS. Default: 1.0.",
    )
    parser.add_argument(
        "--reweighted-policy-weight",
        type=float,
        default=0.5,
        help="Policy weight para samples reweighted. Default: 0.5.",
    )
    parser.add_argument(
        "--sample-limit-per-cycle",
        type=int,
        default=None,
        help="Limite de samples por ciclo. Default: sin ciclos.",
    )
    parser.add_argument(
        "--checkpoint-path",
        default=None,
        help=(
            "Checkpoint ResNet usado por el evaluator. Default: .pt con mayor "
            "step numerico dentro de --checkpoint-dir."
        ),
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="modules/evaluators/resnet/checkpoints",
        help="Directorio donde el learner guarda nuevos checkpoints.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device del evaluator: auto, cpu o cuda. Default: auto.",
    )
    parser.add_argument(
        "--evaluator-batch-size",
        type=int,
        default=32,
        help="Batch maximo del evaluator centralizado. Default: 32.",
    )
    parser.add_argument(
        "--evaluator-batch-wait",
        type=float,
        default=0.01,
        help="Espera maxima para armar batch de inferencia. Default: 0.01.",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=None,
        help="Timeout opcional de requests al evaluator. Default: None.",
    )
    parser.add_argument(
        "--multiprocessing-start-method",
        default="fork",
        help="Metodo multiprocessing para workers>1. Default: fork.",
    )
    parser.add_argument(
        "--train-on-cycle",
        action="store_true",
        help="Entrena la ResNet al cerrar cada ciclo y recarga el evaluator.",
    )
    parser.add_argument(
        "--learner-steps",
        type=int,
        default=None,
        help=(
            "DEPRECATED: los steps se derivan como ceil(samples_del_ciclo / "
            "learner_batch_size)."
        ),
    )
    parser.add_argument(
        "--learner-batch-size",
        type=int,
        default=64,
        help="Batch size del learner si --train-on-cycle. Default: 64.",
    )
    parser.add_argument(
        "--learner-learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate del learner si --train-on-cycle. Default: 1e-4.",
    )
    parser.add_argument(
        "--learner-weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay del learner si --train-on-cycle. Default: 1e-4.",
    )
    parser.add_argument(
        "--learner-value-loss-weight",
        type=float,
        default=1.0,
        help="Peso de value loss si --train-on-cycle. Default: 1.0.",
    )
    parser.add_argument(
        "--learner-policy-loss-weight",
        type=float,
        default=1.0,
        help="Peso global de policy loss si --train-on-cycle. Default: 1.0.",
    )
    parser.add_argument(
        "--reports-dir",
        default=None,
        help="Directorio para logs JSONL. Default: <output-root>/reports.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="ID manual de corrida. Si se pasa, tiene prioridad sobre --run-prefix.",
    )
    parser.add_argument(
        "--run-prefix",
        default="train_gpu_mid",
        help="Prefijo del ID correlativo automatico. Default: train_gpu_mid.",
    )
    parser.add_argument(
        "--disable-report-logging",
        action="store_true",
        help="Desactiva logs persistentes en JSONL.",
    )
    return parser.parse_args()


def list_stock_trajectory_ids(source_path: str | Path) -> list[str]:
    buffer = TrajectoryBuffer(source_path, mode="r")
    return [
        trajectory_id
        for trajectory_id in buffer.list_ids()
        if str(trajectory_id).startswith("stock_")
    ]


def select_source_ids(
    source_ids: list[str],
    n_trajectories: int | None,
    shuffle: bool,
    seed: int | None,
) -> list[str]:
    ids = list(source_ids)
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(ids)

    if n_trajectories is not None:
        ids = ids[: int(n_trajectories)]

    return ids


def parse_checkpoint_step(path: str | Path) -> int:
    match = re.search(r"_(\d+)\.pt$", Path(path).name)
    if match is None:
        return -1
    return int(match.group(1))


def resolve_checkpoint_path(
    checkpoint_path: str | Path | None,
    checkpoint_dir: str | Path,
) -> Path:
    if checkpoint_path is not None:
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"No existe el checkpoint: {path}")
        return path

    candidates = list(Path(checkpoint_dir).glob("*.pt"))
    if not candidates:
        raise FileNotFoundError(
            f"No se encontraron checkpoints .pt en {checkpoint_dir}."
        )

    return max(
        candidates,
        key=lambda path: (parse_checkpoint_step(path), path.stat().st_mtime),
    )


def build_evaluator_for_single_worker(
    source_path: Path,
    source_ids: list[str],
    checkpoint_path: str | Path,
    device: str,
) -> ResNetStateEvaluator:
    if not source_ids:
        raise ValueError("No hay trayectorias fuente seleccionadas.")

    record = TrajectoryBuffer(source_path, mode="r").load(source_ids[0])
    setup = ProblemSetup(**record.problem_setup)
    return ResNetStateEvaluator(
        setup=setup,
        checkpoint_path=checkpoint_path,
        device=device,
    )


def main() -> None:
    args = parse_args()
    paths = create_dataset_buffer_layout(args.output_root)

    source_path = Path(args.source_path) if args.source_path else paths.stock_trajectories
    sample_path = Path(args.sample_path) if args.sample_path else paths.samples
    reports_dir = Path(args.reports_dir) if args.reports_dir else paths.reports
    logger = None
    if not args.disable_report_logging:
        logger = MCTSGenerationRunLogger(
            reports_dir=reports_dir,
            run_id=args.run_id,
            run_prefix=args.run_prefix,
        )

    source_ids = list_stock_trajectory_ids(source_path)
    selected_ids = select_source_ids(
        source_ids=source_ids,
        n_trajectories=args.n_trajectories,
        shuffle=args.shuffle,
        seed=args.seed,
    )
    if not selected_ids:
        if logger is not None:
            logger.log_run(
                status="completed",
                args=args,
                source_path=source_path,
                sample_path=sample_path,
                source_trajectory_count=len(source_ids),
                selected_trajectory_count=0,
                report=None,
            )
        print(f"[mcts_generation] source={source_path}", flush=True)
        print(f"[mcts_generation] samples={sample_path}", flush=True)
        print("[mcts_generation] selected_trajectories=0", flush=True)
        print("[mcts_generation] done", flush=True)
        print("completed_jobs=0", flush=True)
        print("failed_jobs=0", flush=True)
        print("generated_trajectories=0", flush=True)
        print("saved_samples=0", flush=True)
        print("used_mcts_jobs=0", flush=True)
        print("reweighted_jobs=0", flush=True)
        print("cycles=0", flush=True)
        return

    checkpoint_path = resolve_checkpoint_path(
        checkpoint_path=args.checkpoint_path,
        checkpoint_dir=args.checkpoint_dir,
    )
    args.checkpoint_path = str(checkpoint_path)

    mcts_config = MCTSConfig(
        num_simulations=args.mcts_simulations,
        c_puct=args.c_puct,
        mode=MCTSMode.INFERENCE,
        random_seed=args.seed,
    )
    generation_config = MCTSGenerationConfig(
        p_mcts=args.p_mcts,
        start_mode=MCTSStartMode(args.start_mode),
        max_seed_states=args.max_seed_states,
        seed_state_probability=args.seed_state_probability,
        mcts_config=mcts_config,
        tail_window_size=args.tail_window_size,
        mcts_policy_weight=args.mcts_policy_weight,
        reweighted_policy_config=ReweightedPolicyConfig(
            policy_weight=args.reweighted_policy_weight,
        ),
    )

    centralized_config = CentralizedEvaluatorConfig(
        checkpoint_path=checkpoint_path,
        device=args.device,
        max_batch_size=args.evaluator_batch_size,
        batch_wait_s=args.evaluator_batch_wait,
        request_timeout_s=args.request_timeout,
    )

    evaluator = None
    if args.workers == 1:
        evaluator = build_evaluator_for_single_worker(
            source_path=source_path,
            source_ids=selected_ids,
            checkpoint_path=checkpoint_path,
            device=args.device,
        )

    current_checkpoint_path = checkpoint_path

    def on_cycle_ready(cycle_report):
        nonlocal current_checkpoint_path
        print(f"[mcts_generation] cycle_ready={cycle_report}", flush=True)
        if not args.train_on_cycle:
            if logger is not None:
                logger.log_cycle(cycle_report=cycle_report, learner_report=None)
            return None
        if cycle_report.saved_samples <= 0:
            if logger is not None:
                logger.log_cycle(cycle_report=cycle_report, learner_report=None)
            return None

        learner = ResNetSampleLearner(
            ResNetLearnerConfig(
                sample_buffer_path=sample_path,
                checkpoint_path=current_checkpoint_path,
                checkpoint_dir=args.checkpoint_dir,
                device=args.device,
                batch_size=args.learner_batch_size,
                sample_start_index=cycle_report.sample_start_index,
                sample_end_index=cycle_report.sample_end_index,
                learning_rate=args.learner_learning_rate,
                weight_decay=args.learner_weight_decay,
                value_loss_weight=args.learner_value_loss_weight,
                policy_loss_weight=args.learner_policy_loss_weight,
                seed=args.seed,
            )
        )
        learner_report = learner.train()
        if logger is not None:
            logger.log_cycle(
                cycle_report=cycle_report,
                learner_report=learner_report,
            )
            logger.log_learner_steps(
                cycle_index=cycle_report.cycle_index,
                learner_report=learner_report,
            )
        current_checkpoint_path = Path(learner_report.checkpoint_path)
        last_metrics = learner_report.metrics[-1]
        print(
            "[mcts_generation] learner_done "
            f"checkpoint={learner_report.checkpoint_path} "
            f"global_step={learner_report.global_step} "
            f"trained_samples={learner_report.sample_count} "
            f"trained_steps={learner_report.trained_steps} "
            f"last_batch_size={learner_report.last_batch_size} "
            f"loss={last_metrics.loss:.6f} "
            f"policy_loss={last_metrics.policy_loss:.6f} "
            f"value_loss={last_metrics.value_loss:.6f} "
            f"training_seconds={learner_report.training_wall_seconds:.2f} "
            f"zarr_read_seconds={learner_report.zarr_read_total_seconds:.2f} "
            f"encoding_seconds={learner_report.encoding_total_seconds:.2f} "
            f"optimization_seconds={learner_report.optimization_total_seconds:.2f} "
            f"checkpoint_seconds={learner_report.checkpoint_save_total_seconds:.2f} "
            f"samples_per_second={learner_report.samples_per_training_second:.2f}",
            flush=True,
        )
        return current_checkpoint_path

    orchestrator = MCTSGenerationOrchestrator(
        config=MCTSOrchestratorConfig(
            source_buffer_path=source_path,
            sample_buffer_path=sample_path,
            n_workers=args.workers,
            overwrite_samples=args.overwrite_samples,
            print_progress=True,
            progress_interval=args.progress_interval,
            centralized_evaluator_config=centralized_config,
            multiprocessing_start_method=args.multiprocessing_start_method,
            sample_limit_per_cycle=args.sample_limit_per_cycle,
        ),
        generation_config=generation_config,
        evaluator=evaluator,
        on_cycle_ready=on_cycle_ready,
    )
    jobs = build_mcts_generation_jobs(
        source_trajectory_ids=selected_ids,
        seed=args.seed,
    )

    print(f"[mcts_generation] source={source_path}", flush=True)
    print(f"[mcts_generation] samples={sample_path}", flush=True)
    print(f"[mcts_generation] checkpoint={checkpoint_path}", flush=True)
    print(f"[mcts_generation] workers={args.workers}", flush=True)
    print(f"[mcts_generation] source_trajectories={len(source_ids)}", flush=True)
    print(f"[mcts_generation] selected_trajectories={len(selected_ids)}", flush=True)
    print(f"[mcts_generation] p_mcts={args.p_mcts}", flush=True)
    print(f"[mcts_generation] start_mode={args.start_mode}", flush=True)
    print(f"[mcts_generation] train_on_cycle={args.train_on_cycle}", flush=True)
    if args.learner_steps is not None:
        print(
            "[mcts_generation] warning=--learner-steps is deprecated and ignored; "
            "steps are derived from the cycle sample count",
            flush=True,
        )
    if logger is not None:
        print(f"[mcts_generation] run_id={logger.run_id}", flush=True)
        print(f"[mcts_generation] reports_dir={logger.reports_dir}", flush=True)

    report = orchestrator.run(jobs)
    if logger is not None:
        logger.log_run(
            status="completed" if not report.errors else "completed_with_errors",
            args=args,
            source_path=source_path,
            sample_path=sample_path,
            source_trajectory_count=len(source_ids),
            selected_trajectory_count=len(selected_ids),
            report=report,
        )

    print("[mcts_generation] done", flush=True)
    print(f"completed_jobs={report.completed_jobs}", flush=True)
    print(f"failed_jobs={report.failed_jobs}", flush=True)
    print(f"generated_trajectories={report.generated_trajectories}", flush=True)
    print(f"saved_samples={report.saved_samples}", flush=True)
    print(f"used_mcts_jobs={report.used_mcts_jobs}", flush=True)
    print(f"reweighted_jobs={report.reweighted_jobs}", flush=True)
    print(f"cycles={len(report.cycle_reports)}", flush=True)

    if report.errors:
        print("errors:", flush=True)
        for error in report.errors[:10]:
            print(f"- {error}", flush=True)


class MCTSGenerationRunLogger:
    runs_filename = "mcts_generation_runs.jsonl"
    cycles_filename = "mcts_generation_cycles.jsonl"
    learner_steps_filename = "mcts_generation_learner_steps.jsonl"
    sequences_filename = "run_sequences.json"
    sequence_lock_filename = ".run_sequences.lock"

    def __init__(
        self,
        reports_dir: str | Path,
        run_id: str | None = None,
        run_prefix: str = "train_gpu_mid",
    ) -> None:
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = str(
            run_id
            or self._next_correlative_run_id(
                reports_dir=self.reports_dir,
                run_prefix=run_prefix,
            )
        )

    def log_cycle(
        self,
        cycle_report,
        learner_report,
    ) -> None:
        payload = {
            "event": "cycle",
            "run_id": self.run_id,
            "cycle_id": self._cycle_id(cycle_report.cycle_index),
            "created_at": self._now(),
            "cycle": asdict(cycle_report),
            "learner": self._learner_summary(learner_report),
        }
        self._append_jsonl(self.reports_dir / self.cycles_filename, payload)

    def log_learner_steps(
        self,
        cycle_index: int,
        learner_report,
    ) -> None:
        for metric in learner_report.metrics:
            payload = {
                "event": "learner_step",
                "run_id": self.run_id,
                "cycle_id": self._cycle_id(cycle_index),
                "created_at": self._now(),
                "cycle_index": int(cycle_index),
                "checkpoint_path": str(learner_report.checkpoint_path),
                "sample_count": int(learner_report.sample_count),
                "sample_start_index": int(learner_report.sample_start_index),
                "sample_end_index": int(learner_report.sample_end_index),
                "last_batch_size": int(learner_report.last_batch_size),
                "metric": asdict(metric),
            }
            self._append_jsonl(
                self.reports_dir / self.learner_steps_filename,
                payload,
            )

    def log_run(
        self,
        status: str,
        args: argparse.Namespace,
        source_path: str | Path,
        sample_path: str | Path,
        source_trajectory_count: int,
        selected_trajectory_count: int,
        report,
    ) -> None:
        payload = {
            "event": "run",
            "run_id": self.run_id,
            "created_at": self._now(),
            "status": str(status),
            "source_path": str(source_path),
            "sample_path": str(sample_path),
            "source_trajectory_count": int(source_trajectory_count),
            "selected_trajectory_count": int(selected_trajectory_count),
            "args": self._args_to_dict(args),
            "report": self._orchestrator_summary(report),
        }
        self._append_jsonl(self.reports_dir / self.runs_filename, payload)

    @classmethod
    def _learner_summary(cls, learner_report) -> dict[str, Any] | None:
        if learner_report is None:
            return None

        last_metric = learner_report.metrics[-1] if learner_report.metrics else None
        return {
            "checkpoint_path": str(learner_report.checkpoint_path),
            "global_step": int(learner_report.global_step),
            "trained_steps": int(learner_report.trained_steps),
            "sample_count": int(learner_report.sample_count),
            "sample_start_index": int(learner_report.sample_start_index),
            "sample_end_index": int(learner_report.sample_end_index),
            "last_batch_size": int(learner_report.last_batch_size),
            "training_wall_seconds": float(
                learner_report.training_wall_seconds
            ),
            "zarr_read_total_seconds": float(
                learner_report.zarr_read_total_seconds
            ),
            "encoding_total_seconds": float(
                learner_report.encoding_total_seconds
            ),
            "optimization_total_seconds": float(
                learner_report.optimization_total_seconds
            ),
            "checkpoint_save_total_seconds": float(
                learner_report.checkpoint_save_total_seconds
            ),
            "samples_per_training_second": float(
                learner_report.samples_per_training_second
            ),
            "last_metric": asdict(last_metric) if last_metric is not None else None,
        }

    @classmethod
    def _orchestrator_summary(cls, report) -> dict[str, Any] | None:
        if report is None:
            return None

        return {
            "source_buffer_path": str(report.source_buffer_path),
            "sample_buffer_path": str(report.sample_buffer_path),
            "total_jobs": int(report.total_jobs),
            "completed_jobs": int(report.completed_jobs),
            "failed_jobs": int(report.failed_jobs),
            "generated_trajectories": int(report.generated_trajectories),
            "saved_samples": int(report.saved_samples),
            "used_mcts_jobs": int(report.used_mcts_jobs),
            "reweighted_jobs": int(report.reweighted_jobs),
            "cycle_count": len(report.cycle_reports),
            "errors": list(report.errors),
        }

    @staticmethod
    def _args_to_dict(args: argparse.Namespace) -> dict[str, Any]:
        return {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        }

    @staticmethod
    def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(payload, sort_keys=True, default=str))
            file.write("\n")

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _cycle_id(self, cycle_index: int) -> str:
        return f"{self.run_id}_cycle_{int(cycle_index):03d}"

    @classmethod
    def _next_correlative_run_id(
        cls,
        reports_dir: Path,
        run_prefix: str,
    ) -> str:
        prefix = cls._normalize_run_prefix(run_prefix)
        lock_path = reports_dir / cls.sequence_lock_filename

        with lock_path.open("a+", encoding="utf-8") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            sequences_path = reports_dir / cls.sequences_filename
            sequences = cls._read_sequences(sequences_path)
            logged_max = cls._max_logged_sequence(
                reports_dir / cls.runs_filename,
                prefix=prefix,
            )
            next_sequence = max(int(sequences.get(prefix, 0)), logged_max) + 1
            sequences[prefix] = next_sequence
            cls._write_sequences_atomic(sequences_path, sequences)
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

        return f"{prefix}_{next_sequence:03d}"

    @staticmethod
    def _normalize_run_prefix(run_prefix: str) -> str:
        prefix = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(run_prefix).strip())
        prefix = prefix.strip("_")
        if not prefix:
            raise ValueError("--run-prefix no puede quedar vacio.")
        return prefix

    @staticmethod
    def _read_sequences(path: Path) -> dict[str, int]:
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
        return {
            str(key): int(value)
            for key, value in data.items()
            if isinstance(value, int) and value >= 0
        }

    @staticmethod
    def _write_sequences_atomic(path: Path, sequences: dict[str, int]) -> None:
        temporary_path = path.with_suffix(path.suffix + ".tmp")
        temporary_path.write_text(
            json.dumps(sequences, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_path, path)

    @staticmethod
    def _max_logged_sequence(path: Path, prefix: str) -> int:
        if not path.exists():
            return 0

        pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)$")
        maximum = 0
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                try:
                    run_id = str(json.loads(line).get("run_id", ""))
                except json.JSONDecodeError:
                    continue
                match = pattern.match(run_id)
                if match is not None:
                    maximum = max(maximum, int(match.group(1)))
        return maximum


if __name__ == "__main__":
    main()
