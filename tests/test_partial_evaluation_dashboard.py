from scripts.partial_evaluation_dashboard import (
    merge_partial_evaluation_metadata,
    render_partial_evaluation_dashboard,
)


def test_merge_partial_evaluation_metadata_joins_source_trajectory() -> None:
    dataset = {
        "trajectories": [
            {
                "trajectory_id": "partial_000001",
                "resources_total": 7,
                "initial_demand_total": 500,
                "has_expansion_mode": True,
                "stock_was_reduced": True,
                "metadata": {"pipeline": "raw_noise_stock"},
            }
        ]
    }
    evaluation = {
        "trajectories": [
            {
                "trajectory_id": "partial_mcts_000001",
                "source_trajectory_id": "partial_000001",
                "final_reward": 0.25,
            }
        ]
    }

    merged = merge_partial_evaluation_metadata(dataset, evaluation)

    row = merged["trajectories"][0]
    assert row["source_resources_total"] == 7
    assert row["source_initial_demand_total"] == 500
    assert row["source_has_expansion_mode"] is True
    assert row["source_metadata"]["pipeline"] == "raw_noise_stock"


def test_render_partial_dashboard_contains_requested_sections() -> None:
    html = render_partial_evaluation_dashboard(
        {
            "generated_at": "2026-06-08T12:00:00",
            "dataset": {"trajectories": []},
            "evaluation": {"trajectories": []},
        }
    )

    assert "Descripcion del test dataset" in html
    assert "Resultados MCTS + ResNet" in html
    assert "Better or equal rate por tail y modelo" in html
    assert "Detalles" in html
