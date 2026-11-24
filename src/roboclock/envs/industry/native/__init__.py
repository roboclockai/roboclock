"""Native Industry scenes, privileged controllers and their recorded runner loop.

One package replaces the ten frozen per-task source profiles that used to run in
separate processes with their own ``roboclock`` tree on PYTHONPATH. It imports the
main package's simulator helpers and laboratory appearance instead of shadowing
them. ``catalog.TASKS`` lists each task's scene/controller and recorded loop
settings; ``runner.py`` is the standalone recording path and
``roboclock.envs.industry.env.IndustryEnv`` the benchmark environment. Importing
this package does not start Isaac Sim.
"""
