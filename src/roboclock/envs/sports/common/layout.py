"""Seeded metric layouts; physical geometry is independent of timing difficulty."""

from dataclasses import dataclass, field

import numpy as np

COLORS = [
    [0.85, 0.12, 0.12],
    [0.10, 0.38, 0.90],
    [0.15, 0.70, 0.30],
    [0.95, 0.70, 0.08],
    [0.62, 0.18, 0.78],
]


@dataclass
class Body:
    name: str
    position: list[float]
    radius: float = 0.016
    height: float = 0.055
    mass: float = 0.045
    color: list[float] = field(default_factory=lambda: [0.8, 0.5, 0.1])
    shape: str = "cylinder"
    label: str = ""


@dataclass
class Control:
    name: str
    position: list[float]
    kind: str = "press"
    radius: float = 0.030
    height: float = 0.012
    travel: float = 0.008
    threshold: float = 0.003
    color: list[float] = field(default_factory=lambda: [0.2, 0.7, 0.3])


@dataclass
class Layout:
    bodies: dict = field(default_factory=dict)
    controls: dict = field(default_factory=dict)
    targets: dict = field(default_factory=dict)
    fixtures: list = field(default_factory=list)
    jitter: list = field(default_factory=list)
    friction: float = 0.8
    light_intensity: float = 700.0


def make_layout(task_id, seed, config):
    rng = np.random.default_rng(np.random.SeedSequence([seed, int(task_id), 719]))
    scene = config["scene"]
    h = scene["table_height"]
    jitter = rng.uniform(-scene["spatial_jitter"], scene["spatial_jitter"], 2)
    layout = Layout(
        jitter=jitter.tolist(),
        friction=float(rng.uniform(*scene["friction_range"])),
        light_intensity=float(rng.uniform(600, 800)),
    )

    def pos(x, y, z):
        return [float(x + jitter[0]), float(y + jitter[1]), float(z)]

    def body(name, x, y, r=0.016, height=0.055, shape="cylinder", color=None, label=""):
        layout.bodies[name] = Body(
            name,
            pos(x, y, h + height / 2 + 0.002),
            r,
            height,
            float(rng.uniform(*scene["mass_range"])),
            color or COLORS[0],
            shape,
            label,
        )

    def control(name, x, y, kind="press", radius=0.03, color=None):
        z = h + 0.050
        travel = (
            0.008
            if kind == "press"
            else 0.055
            if kind == "pull"
            else 1.2
            if kind == "twist"
            else 0.055
        )
        threshold = (
            0.003
            if kind in ("press", "mole")
            else config["tasks"]["36"]["pull_distance"]
            if kind == "pull"
            else config["tasks"]["36"]["twist_radians"]
        )
        layout.controls[name] = Control(
            name,
            pos(x, y, z),
            kind,
            radius,
            0.020 if kind in ("pull", "twist") else 0.012,
            travel,
            threshold,
            color or COLORS[0],
        )

    def fixture(name, x, y, z, size, color):
        layout.fixtures.append(
            {"name": name, "position": pos(x, y, z), "size": size, "color": color}
        )

    if task_id == "32":
        fixture("rhythm_console", 0.49, 0, h + 0.016, [0.22, 0.46, 0.032], [0.035, 0.045, 0.065])
        for i in range(3):
            control(f"pad_{i}", 0.46, (i - 1) * 0.14, radius=0.042, color=COLORS[i])
    elif task_id == "33":
        body(
            "dumbbell",
            0.43,
            0,
            r=0.015,
            height=0.074436,
            shape="dumbbell",
            color=[0.15, 0.18, 0.22],
            label="LIFT",
        )
        layout.bodies["dumbbell"].position[2] = h + 0.012 + 0.074436 / 2 + 0.002
        layout.bodies["dumbbell"].mass = 2.27
        fixture("platform", 0.43, 0, h + 0.006, [0.19, 0.23, 0.012], [0.2, 0.3, 0.38])
        layout.targets["dumbbell"] = layout.bodies["dumbbell"].position.copy()
    elif task_id == "34":
        # A raised cabinet leaves room for fully retracted heads above the table.
        fixture("mole_floor", 0.52, 0, h + 0.004, [0.36, 0.36, 0.008], [0.045, 0.06, 0.08])
        for axis in range(2):
            for sign in (-1, 1):
                xy = [0.52, 0.0]
                xy[axis] += sign * 0.185
                size = [0.01, 0.38, 0.10] if axis == 0 else [0.36, 0.01, 0.10]
                fixture(f"mole_wall_{axis}_{sign}", *xy, h + 0.05, size, [0.08, 0.19, 0.23])
        for i in range(9):
            control(
                f"mole_{i}",
                0.40 + (i // 3) * 0.12,
                (i % 3 - 1) * 0.12,
                kind="mole",
                radius=0.025,
                color=COLORS[2],
            )
            mole = layout.controls[f"mole_{i}"]
            mole.position[2] = h + 0.115
            mole.height, mole.travel = 0.050, 0.080
    elif task_id == "35":
        control("buzzer", 0.45, 0, radius=0.04, color=[0.8, 0.12, 0.12])
    elif task_id == "36":
        control("press", 0.47, -0.15)
        control("pull", 0.47, 0, kind="pull", radius=0.014, color=COLORS[1])
        control("twist", 0.47, 0.15, kind="twist", radius=0.014, color=COLORS[3])
    elif task_id == "37":
        body(
            "puck",
            0.44,
            -0.10,
            r=0.027,
            height=0.016,
            color=[0.06, 0.08, 0.1],
            shape="puck",
            label="PUCK",
        )
        layout.targets["puck"] = pos(0.66, -0.10, h + 0.008)
        fixture("goal_mark", 0.66, -0.10, h + 0.0003, [0.12, 0.12, 0.0006], [0.13, 0.45, 0.23])
    elif task_id == "38":
        # Open loading side keeps the gripper clear of the adjacent target deck.
        for x in (0.33,):
            fixture(
                f"pin_rack_side_{x}", x, -0.05, h + 0.004, [0.005, 0.42, 0.008], [0.16, 0.27, 0.34]
            )
        for y in (-0.232, 0.132):
            fixture(
                f"pin_rack_end_{y}", 0.4085, y, h + 0.004, [0.122, 0.005, 0.008], [0.16, 0.27, 0.34]
            )
        fixture("spot_deck", 0.575, 0, h + 0.0002, [0.255, 0.265, 0.0004], [0.18, 0.28, 0.30])
        for i in range(10):
            body(
                f"pin_{i}",
                0.38 + 0.065 * (i // 5),
                -0.20 + 0.075 * (i % 5),
                r=0.013,
                height=0.065,
                color=[0.92, 0.92, 0.85],
                label=str(i + 1),
            )
        slot = 0
        for row in range(4):
            for j in range(row + 1):
                layout.targets[f"pin_{slot}"] = pos(
                    0.49 + row * 0.057, (j - row / 2) * 0.063, h + 0.0325
                )
                slot += 1
    elif task_id == "39":
        for i in range(5):
            body(
                f"cup_{i}",
                0.42,
                (i - 2) * 0.085,
                r=0.016,
                height=0.050,
                shape="cup",
                color=COLORS[i],
                label=str(i + 1),
            )
            layout.targets[f"slot_{i}"] = pos(0.56, (i - 2) * 0.085, h + 0.025)
        control("bell", 0.45, -0.29, radius=0.028, color=[0.85, 0.65, 0.15])
    elif task_id == "40":
        for x in (0.353, 0.572):
            fixture(
                f"basket_side_{x}", x, -0.1375, h + 0.004, [0.004, 0.225, 0.008], [0.18, 0.27, 0.32]
            )
        for y in (-0.247, -0.028):
            fixture(
                f"basket_end_{y}", 0.4625, y, h + 0.004, [0.223, 0.004, 0.008], [0.18, 0.27, 0.32]
            )
        for i in range(config["tasks"]["40"]["balls"]):
            body(
                f"ball_{i}",
                0.38 + 0.055 * (i // 4),
                -0.22 + 0.055 * (i % 4),
                r=0.017,
                height=0.034,
                shape="sphere",
                color=[0.75, 0.9, 0.15],
            )
        center = pos(0.56, 0.15, h + 0.12)
        layout.targets["hopper"] = center
        fixture("hopper_floor", 0.56, 0.15, h + 0.006, [0.18, 0.18, 0.012], [0.18, 0.23, 0.28])
        for axis in range(2):
            for sign in (-1, 1):
                x, y = 0.56, 0.15
                if axis == 0:
                    x += sign * 0.095
                else:
                    y += sign * 0.095
                size = [0.012, 0.20, 0.13] if axis == 0 else [0.20, 0.012, 0.13]
                fixture(f"hopper_wall_{axis}_{sign}", x, y, h + 0.065, size, [0.24, 0.32, 0.36])
        # Shallow separate source pockets stop round balls rolling before grasping.
        for i, b in enumerate(layout.bodies.values()):
            for axis in range(2):
                for sign in (-1, 1):
                    p = np.array(b.position)
                    p[axis] += sign * 0.021
                    p[2] = h + 0.003
                    size = [0.004, 0.044, 0.006] if axis == 0 else [0.044, 0.004, 0.006]
                    layout.fixtures.append(
                        {
                            "name": f"pocket_{i}_{axis}_{sign}",
                            "position": p.tolist(),
                            "size": size,
                            "color": [0.2, 0.24, 0.29],
                        }
                    )
    return layout
