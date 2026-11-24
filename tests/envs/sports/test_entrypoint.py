import sys
from types import SimpleNamespace

from roboclock.envs.sports import demo


def test_unified_c1_runs_racking_instead_of_historical_clearance(monkeypatch):
    invoked = []
    monkeypatch.setattr(
        sys, "argv", ["fixed-demo", "--task", "C1", "--seed", "7", "--output", "/tmp/test-rack"]
    )
    monkeypatch.setattr(
        demo.importlib,
        "import_module",
        lambda name: SimpleNamespace(main=lambda: invoked.append((name, sys.argv.copy()))),
    )
    demo.main()
    assert invoked == [
        (
            "roboclock.envs.sports.pool_rack.demo",
            [
                "roboclock.envs.sports.pool_rack.demo",
                "--embodiment",
                "dual_arx_x5",
                "--seed",
                "7",
                "--output",
                "/tmp/test-rack",
            ],
        )
    ]
