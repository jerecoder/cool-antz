from ant_byte_env.research.markdown import research_experiment_markdown


def test_research_experiment_markdown_renders_course_relevant_sections() -> None:
    markdown = research_experiment_markdown(
        {
            "id": "DISTANCE_CAP4",
            "title": "Distance shaped four ant policy",
            "family": "combined_reward_capacity",
            "mode": "forage_curriculum",
            "run_dir": "runs/autoresearch/DISTANCE_CAP4",
            "source_checkpoint": "source.pkl",
            "hypothesis": "Four ants improve sparse exploration.",
            "intervention": "Use distance shaping and four ants.",
            "target": {"baseline": "single ant weak 25x25 result"},
            "success_signal": "held-out 25x25 delivery",
            "evaluation": {"sampled_episodes": 8},
            "report_notes": "Good presentation result.",
            "resolved_args": {"num_ants": 4},
            "stages": [{"name": "25x25"}],
        }
    )

    assert markdown.startswith("# DISTANCE_CAP4: Distance shaped four ant policy")
    assert "Source checkpoint: `source.pkl`" in markdown
    assert "## Evaluation Gate" in markdown
    assert '"sampled_episodes": 8' in markdown
    assert "## Stage Schedule" in markdown
