from __future__ import annotations

import random
from dataclasses import dataclass

from .raw_worker import RawDemandTrajectoryWorker
from .schemas import GenerationJob, GenerationWorkerResult
from .stock_worker import StockAdjustmentConfig, StockAdjustmentTrajectoryWorker


@dataclass
class RawStockTrajectoryWorker:
    """Genera una trayectoria raw con ruido y luego aplica ajuste de stock."""

    raw_worker: RawDemandTrajectoryWorker
    stock_config: StockAdjustmentConfig = StockAdjustmentConfig()
    trajectory_id_prefix: str = "stock"

    worker_type: str = "raw_stock"

    def run(self, job: GenerationJob) -> GenerationWorkerResult:
        rng = random.Random(int(job.seed))
        raw_seed = rng.randint(0, 2**31 - 1)
        stock_seed = rng.randint(0, 2**31 - 1)

        raw_job = GenerationJob(
            job_id=job.job_id,
            seed=raw_seed,
            payload=dict(job.payload),
        )
        raw_result = self.raw_worker.run(raw_job)
        generated_raw = raw_result.trajectories[0]

        stock_worker = StockAdjustmentTrajectoryWorker(
            source_buffer_path=None,
            config=self.stock_config,
            trajectory_id_prefix=self.trajectory_id_prefix,
        )
        stock_job = GenerationJob(
            job_id=job.job_id,
            seed=stock_seed,
            payload={"source_trajectory_id": generated_raw.trajectory_id},
        )
        result = stock_worker.run_from_generated(stock_job, generated_raw)

        raw_metadata = {
            f"raw_{key}": value
            for key, value in generated_raw.metadata.items()
            if key not in {"final_reward", "final_value", "trajectory_length"}
        }
        for generated in result.trajectories:
            generated.metadata = {
                **raw_metadata,
                **generated.metadata,
                "pipeline": "raw_noise_stock",
                "pipeline_seed": int(job.seed),
                "raw_seed": int(raw_seed),
                "stock_seed": int(stock_seed),
            }
        result.metadata = dict(result.trajectories[0].metadata)
        result.worker_type = self.worker_type
        return result
