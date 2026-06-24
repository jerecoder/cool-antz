from ant_byte_env.research import scoring


def test_summary_score_prefers_evaluation_metrics() -> None:
    summary = {
        "evaluation": {
            "sampled": {
                "eval_mean_episode_return": 10.0,
                "eval_mean_delivered_food": 20.0,
                "eval_mean_delivered_fraction": 0.5,
            }
        },
        "curriculum": {
            "stage_metrics": [
                {
                    "stage_name": "25x25",
                    "episode_return": 1.0,
                    "delivery_events": 1.0,
                    "final_mean_remaining_food": 10.0,
                }
            ]
        },
    }

    assert scoring.summary_score(summary) == 10.65


def test_target_stage_metrics_uses_last_matching_stage_row() -> None:
    summary = {
        "curriculum": {
            "stage_metrics": [
                {"stage_name": "25x25", "episode_return": 1.0},
                {"stage_name": "20x20", "episode_return": 9.0},
                {"stage_name": "25x25", "episode_return": 2.0},
            ]
        }
    }

    assert scoring.target_stage_metrics(summary, target_stage="25x25") == {
        "stage_name": "25x25",
        "episode_return": 2.0,
    }


def test_flatten_evaluation_metrics_removes_eval_prefix() -> None:
    assert scoring.flatten_evaluation_metrics(
        {
            "sampled": {
                "eval_mean_episode_return": 3.0,
                "eval_success_rate": 0.5,
                "label": "ignore",
            },
            "score": 4.0,
        }
    ) == {
        "evaluation/sampled/mean_episode_return": 3.0,
        "evaluation/sampled/success_rate": 0.5,
        "evaluation/score": 4.0,
    }


def test_extra_evaluation_summary_excludes_primary_modes() -> None:
    assert scoring.extra_evaluation_summary(
        {
            "deterministic": {"eval_mean_episode_return": 1.0},
            "sampled_move_greedy_write": {
                "eval_mean_episode_return": 2.0,
                "eval_mean_delivered_food": 3.0,
                "eval_mean_delivered_fraction": 0.4,
                "eval_success_rate": 0.5,
                "eval_mean_episode_length": 6.0,
            },
        }
    ) == {
        "sampled_move_greedy_write": {
            "return": 2.0,
            "delivered": 3.0,
            "fraction": 0.4,
            "success_rate": 0.5,
            "episode_length": 6.0,
        }
    }
