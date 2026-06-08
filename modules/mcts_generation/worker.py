from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from modules.mcts.mcts import MCTS
from modules.storage import TrajectoryBuffer
from modules.trajectory_generation import generate_mcts_trajectory
from modules.workforce_engine.engine import WorkforceEngine
from modules.workforce_engine.schemas import ProblemSetup, WorkforceState

from .config import MCTSGenerationConfig
from .policies import build_reweighted_policy
from .schemas import GeneratedSampleTrajectory, MCTSGenerationJob, MCTSWorkerResult
from .seed_selection import select_seed_step_indices


@dataclass
class MCTSGenerationWorker:
    source_buffer_path: str | Path
    config: MCTSGenerationConfig
    evaluator: Any
    trajectory_id_prefix: str = "mcts"

    worker_type: str = "mcts_generation"

    def run(self, job: MCTSGenerationJob) -> MCTSWorkerResult:
        rng = random.Random(int(job.seed))
        record = TrajectoryBuffer(self.source_buffer_path, mode="r").load(
            job.source_trajectory_id
        )
        setup = ProblemSetup(**record.problem_setup)
        engine = WorkforceEngine(setup)
        self._prepare_evaluator_for_setup(setup)

        if rng.random() < float(self.config.p_mcts):
            trajectories = self._generate_mcts_trajectories(
                job=job,
                record=record,
                setup=setup,
                engine=engine,
                rng=rng,
            )
            used_mcts = True
        else:
            trajectories = [
                self._build_reweighted_trajectory(
                    job=job,
                    record=record,
                    setup=setup,
                )
            ]
            used_mcts = False

        return MCTSWorkerResult(
            job_id=job.job_id,
            source_trajectory_id=job.source_trajectory_id,
            trajectories=trajectories,
            used_mcts=used_mcts,
            metadata={
                "worker_type": self.worker_type,
                "used_mcts": bool(used_mcts),
                "generated_trajectories": len(trajectories),
            },
        )

    def _generate_mcts_trajectories(
        self,
        job: MCTSGenerationJob,
        record,
        setup: ProblemSetup,
        engine: WorkforceEngine,
        rng: random.Random,
    ) -> list[GeneratedSampleTrajectory]:
        seed_indices = select_seed_step_indices(
            trajectory_length=len(record.samples),
            start_mode=self.config.start_mode,
            max_seed_states=self.config.max_seed_states,
            seed_state_probability=self.config.seed_state_probability,
            rng=rng,
            tail_window_size=self.config.tail_window_size,
        )

        generated: list[GeneratedSampleTrajectory] = []
        for output_index, source_step_index in enumerate(seed_indices):
            source_sample = record.samples[int(source_step_index)]
            initial_state = self._coerce_state(source_sample["state"])
            if hasattr(initial_state, "copy_state"):
                initial_state = initial_state.copy_state()

            if engine.check_terminality(initial_state):
                continue

            mcts = MCTS(
                engine=engine,
                evaluator=self.evaluator,
                config=self.config.mcts_config.model_copy(
                    update={"random_seed": rng.randint(0, 2**31 - 1)}
                ),
            )
            trajectory, final_reward, _ = generate_mcts_trajectory(
                initial_state=initial_state,
                engine=engine,
                mcts=mcts,
                debug=False,
            )
            trajectory_id = (
                f"{self.trajectory_id_prefix}_{job.source_trajectory_id}_"
                f"{output_index:03d}"
            )
            metadata = {
                "sample_source": "mcts",
                "source_trajectory_id": job.source_trajectory_id,
                "source_step_index": int(source_step_index),
                "start_mode": self.config.start_mode.value,
                "tail_window_size": self.config.tail_window_size,
                "policy_weight": float(self.config.mcts_policy_weight),
                "final_reward": float(final_reward),
            }

            generated.append(
                GeneratedSampleTrajectory(
                    trajectory=self._with_sample_metadata(
                        trajectory=trajectory,
                        policy_weight=float(self.config.mcts_policy_weight),
                        metadata=metadata,
                    ),
                    problem_setup=setup,
                    trajectory_id=trajectory_id,
                    metadata=metadata,
                )
            )

        return generated

    def _build_reweighted_trajectory(
        self,
        job: MCTSGenerationJob,
        record,
        setup: ProblemSetup,
    ) -> GeneratedSampleTrajectory:
        policy_weight = float(self.config.reweighted_policy_config.policy_weight)
        trajectory = []
        metadata = {
            "sample_source": "stock_reweighted",
            "source_trajectory_id": job.source_trajectory_id,
            "policy_weight": policy_weight,
            "final_reward": float(record.final_reward),
        }

        for sample in record.samples:
            trajectory.append(
                {
                    "state": sample["state"],
                    "policy": build_reweighted_policy(
                        original_policy=sample["policy"],
                        selected_action_id=int(sample["action_id"]),
                    ),
                    "action_id": int(sample["action_id"]),
                    "value": float(sample["reward"]),
                    "reward": float(sample["reward"]),
                    "policy_weight": policy_weight,
                    "metadata": {
                        **metadata,
                        "source_step_index": int(sample["step_index"]),
                    },
                }
            )

        return GeneratedSampleTrajectory(
            trajectory=trajectory,
            problem_setup=setup,
            trajectory_id=f"reweighted_{job.source_trajectory_id}",
            metadata=metadata,
        )

    @staticmethod
    def _with_sample_metadata(
        trajectory: list[dict[str, Any]],
        policy_weight: float,
        metadata: dict[str, Any],
    ) -> list[dict[str, Any]]:
        output = []
        for step_index, sample in enumerate(trajectory):
            output.append(
                {
                    **sample,
                    "value": float(sample["reward"]),
                    "policy_weight": float(policy_weight),
                    "metadata": {
                        **metadata,
                        "generated_step_index": int(step_index),
                    },
                }
            )
        return output

    @staticmethod
    def _coerce_state(state: Any) -> WorkforceState:
        if isinstance(state, WorkforceState):
            return state
        return WorkforceState(**state)

    def _prepare_evaluator_for_setup(self, setup: ProblemSetup) -> None:
        if hasattr(self.evaluator, "setup"):
            self.evaluator.setup = setup
