from scripts.build_training_dashboard import shared_dashboard_javascript


def test_shared_charts_separate_runs_and_reset_ranges() -> None:
    javascript = shared_dashboard_javascript()

    assert "function resetChartRanges()" in javascript
    assert "const runSeparators = visibleRows.map" in javascript
    assert "previous?.run_id !== row.run_id" in javascript
    assert "${runSeparators}" in javascript
