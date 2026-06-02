from modules.dataset_generation.orchestrator import _add_resource_totals


def test_add_resource_totals_uses_initial_stock() -> None:
    totals = {"mod_4": 0, "mod_6": 0, "mod_8": 0}

    _add_resource_totals(totals, {"initial_stock": [1, 2, 3]})

    assert totals == {"mod_4": 1, "mod_6": 2, "mod_8": 3}


def test_add_resource_totals_falls_back_to_output_stock() -> None:
    totals = {"mod_4": 0, "mod_6": 0, "mod_8": 0}

    _add_resource_totals(totals, {"output_stock": [4, 5, 6]})

    assert totals == {"mod_4": 4, "mod_6": 5, "mod_8": 6}


def test_add_resource_totals_ignores_missing_stock() -> None:
    totals = {"mod_4": 0, "mod_6": 0, "mod_8": 0}

    _add_resource_totals(totals, {})

    assert totals == {"mod_4": 0, "mod_6": 0, "mod_8": 0}
