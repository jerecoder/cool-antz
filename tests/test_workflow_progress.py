from ant_byte_env import notebook_workflows as workflows
from ant_byte_env.workflows import progress


class FakeProgress:
    def __init__(self) -> None:
        self.updates: list[int] = []

    def update(self, value: int) -> None:
        self.updates.append(value)


def test_advance_progress_to_updates_only_forward_delta() -> None:
    bar = FakeProgress()

    assert progress.advance_progress_to(
        bar,
        update_index=5,
        previous_update_index=2,
    ) == 5
    assert progress.advance_progress_to(
        bar,
        update_index=3,
        previous_update_index=5,
    ) == 3

    assert bar.updates == [3, 0]


def test_notebook_workflows_reexports_progress_helpers() -> None:
    assert workflows._advance_progress_to is progress.advance_progress_to
    assert workflows.stage_update_progress is progress.stage_update_progress
