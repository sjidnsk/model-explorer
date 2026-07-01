from __future__ import annotations


def test_path_feedback_selection_api_uses_new_module() -> None:
    from model_explorer.policy.feedback_selection import selected_after_feedback

    assert callable(selected_after_feedback)
