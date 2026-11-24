# Shared configuration templates

`benchmark.yaml` supplies the validated common rendering, motor, contact and scene
schema used by the Dual-X5 environment. Its inherited prototype episode fields
are retained for schema compatibility; they are not the current task definitions.

The current Dual-X5 tasks in `../tasks.json` define their parameters in `<id>_x5/task.py`.

`no_render.yaml` is `benchmark.yaml` with `x5.no_render: true`: the Dual-X5 environment then
creates no cameras and makes no render calls, and every observation carries zero images of the
rendered shapes (RoboDojo rig spec). Use it for headless oracle runs:
`roboclock eval --env household --config src/roboclock/envs/household/configs/no_render.yaml --policy oracle --oracle ...`.
