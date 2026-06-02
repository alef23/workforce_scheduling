from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path


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
        default="modules/evaluators/resnet/checkpoints/workforce_resnet_000.pt",
        help="Checkpoint ResNet usado por el evaluator.",
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
        default=100,
        help="Steps de entrenamiento por ciclo si --train-on-cycle. Default: 100.",
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

    source_ids = list_stock_trajectory_ids(source_path)
    selected_ids = select_source_ids(
        source_ids=source_ids,
        n_trajectories=args.n_trajectories,
        shuffle=args.shuffle,
        seed=args.seed,
    )
    if not selected_ids:
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
        mcts_policy_weight=args.mcts_policy_weight,
        reweighted_policy_config=ReweightedPolicyConfig(
            policy_weight=args.reweighted_policy_weight,
        ),
    )

    centralized_config = CentralizedEvaluatorConfig(
        checkpoint_path=args.checkpoint_path,
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
            checkpoint_path=args.checkpoint_path,
            device=args.device,
        )

    current_checkpoint_path = Path(args.checkpoint_path)

    def on_cycle_ready(cycle_report):
        nonlocal current_checkpoint_path
        print(f"[mcts_generation] cycle_ready={cycle_report}", flush=True)
        if not args.train_on_cycle:
            return None
        if cycle_report.saved_samples <= 0:
            return None

        learner = ResNetSampleLearner(
            ResNetLearnerConfig(
                sample_buffer_path=sample_path,
                checkpoint_path=current_checkpoint_path,
                checkpoint_dir=args.checkpoint_dir,
                device=args.device,
                batch_size=args.learner_batch_size,
                train_steps=args.learner_steps,
                learning_rate=args.learner_learning_rate,
                weight_decay=args.learner_weight_decay,
                value_loss_weight=args.learner_value_loss_weight,
                policy_loss_weight=args.learner_policy_loss_weight,
                seed=args.seed,
            )
        )
        learner_report = learner.train()
        current_checkpoint_path = Path(learner_report.checkpoint_path)
        last_metrics = learner_report.metrics[-1]
        print(
            "[mcts_generation] learner_done "
            f"checkpoint={learner_report.checkpoint_path} "
            f"global_step={learner_report.global_step} "
            f"loss={last_metrics.loss:.6f} "
            f"policy_loss={last_metrics.policy_loss:.6f} "
            f"value_loss={last_metrics.value_loss:.6f}",
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
    print(f"[mcts_generation] workers={args.workers}", flush=True)
    print(f"[mcts_generation] source_trajectories={len(source_ids)}", flush=True)
    print(f"[mcts_generation] selected_trajectories={len(selected_ids)}", flush=True)
    print(f"[mcts_generation] p_mcts={args.p_mcts}", flush=True)
    print(f"[mcts_generation] start_mode={args.start_mode}", flush=True)
    print(f"[mcts_generation] train_on_cycle={args.train_on_cycle}", flush=True)

    report = orchestrator.run(jobs)

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


if __name__ == "__main__":
    main()
