"""Public workflow displays shared by recordings and agent observations."""

from roboclock.envs.sports.arcade.scenes import vertical_label


def shot_display(workflow, time):
    """Expose the active window, without a countdown to the hidden READY event."""
    if time < workflow["shot_start"]:
        return "WAIT FOR READY"
    remaining = workflow["shot_start"] + workflow["shot_clock"] - time
    if remaining <= 0:
        return "SHOT CLOCK / EXPIRED"
    return f"READY / SHOT CLOCK {remaining:04.1f}"


def update_cues(b, task):
    from pxr import Gf, UsdGeom

    c = b.config["workflow"]
    time = b.time
    if b.kind == "A3":
        label = shot_display(c, time)
        if task.basket:
            label = "BASKET / ON TIME" if task.temporal else "BASKET / TOO LATE"
        if label != getattr(b, "cue_label", None):
            vertical_label(b, "/World/ShotClock/Status", label, [0.70, 0.543, 1.445], [0.64, 0.065])
    elif b.kind == "B2":
        update_boxing(b, task)
        return
    else:
        # No countdown: C2 has no in-episode time budget (score_version 2), only card counts.
        display_card = min(task.card, c["cups_required"] - 1)
        label = (
            f"{'L STACK / R ROW' if task.card % 2 == 0 else 'L ROW / R STACK'}"
            f" / CARD {task.card + 1}"
        )
        if task.card >= c["cups_required"]:
            label = f"{task.card}/{c['cups_required']} CARDS"
        if task.done:
            label = f"ROUND COMPLETE / {task.card} CARDS"
        if label != getattr(b, "cue_label", None):
            vertical_label(b, "/World/CupScreen/Status", label, [0.63, 0.433, 1.40], [0.42, 0.035])
        if getattr(b, "shown_card", None) != display_card:
            from roboclock.envs.sports.arena.cup_patterns import icons, layout

            targets, supports = layout(display_card)
            for name, target in targets.items():
                marker = b.stage.GetPrimAtPath(f"/World/Cups/TargetMark{name}")
                op = UsdGeom.Xformable(marker).GetOrderedXformOps()[0]
                op.Set(Gf.Vec3d(float(target[0]), float(target[1]), 0.755))
                image = UsdGeom.Imageable(marker)
                image.MakeVisible() if supports[name] == ["Table"] else image.MakeInvisible()
            for side, dx in [("left", -0.11), ("right", 0.11)]:
                for i in range(3):
                    p = b.stage.GetPrimAtPath(f"/World/Cups/Icon{side}{i}")
                    xf = UsdGeom.Xformable(p)
                    tr = next(
                        op
                        for op in xf.GetOrderedXformOps()
                        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate
                    )
                    ix, iz = icons(display_card, side)[i]
                    x, z = 0.63 + dx + ix, 1.20 + iz
                    tr.Set(Gf.Vec3d(x, 0.432, z))
            b.shown_card = display_card
    b.cue_label = label


def boxing_display(task, time):
    """Public screen content: current cue only, never the future cue schedule."""
    from roboclock.envs.sports.arena.boxing import active_command, next_strike

    c = task.c
    index = active_command(task.commands, time)
    title = "WAIT FOR THE CUE"
    action, timer, fraction, color, lit = "GUARD / WAIT", "DO NOT PUNCH", 0.0, "amber", None
    if index is not None:
        command = task.commands[index]
        title = f"{index + 1:02d} / {command['name']}"
        if not command["hands"]:
            action = "HOLD / NO PUNCH"
            left = max(0, command["end"] - time)
            timer, fraction = f"HOLD {left:03.1f}s", left / (command["end"] - command["time"])
    expected = next_strike(task, time)
    if expected is not None:
        _, count, hand, cue = expected
        remaining = c["cue_response"] - (time - cue)
        action = f"GO {hand.upper()} / {count + 1}"
        timer = f"HIT IN {max(0, remaining):03.1f}s" if remaining >= 0 else "HIT WINDOW EXPIRED"
        fraction, color, lit = (
            max(0, remaining / c["cue_response"]),
            "green" if remaining >= 0 else "red",
            hand,
        )
    if task.pending_guard:
        hand, hit_time = next(iter(task.pending_guard.items()))
        remaining = c["guard_deadline"] - (time - hit_time)
        action, timer = f"GUARD {hand.upper()}", f"RETURN IN {max(0, remaining):03.1f}s"
        if remaining < 0:
            timer = "GUARD WINDOW EXPIRED"
        fraction, color, lit = (
            max(0, remaining / c["guard_deadline"]),
            "cyan" if remaining >= 0 else "red",
            None,
        )
    last = "HIT LIMIT %.2fs / GUARD %.2fs" % (c["cue_response"], c["guard_deadline"])
    if task.hits:
        hit = task.hits[-1]
        reaction = next(
            (
                r
                for r in task.reactions
                if r["command"] == hit["command"] and r["strike"] == hit["strike"]
            ),
            None,
        )
        last = f"HIT {hit['latency']:.2f}s / {c['cue_response']:.2f}s"
        if reaction is not None:
            last = f"START {reaction['latency']:.2f}s / " + last
    if task.done:
        title = "ROUND COMPLETE"
        action = "ON TIME" if task.result()["success"] else "SEE TIMING RESULT"
        timer, fraction, color, lit = f"{len(task.hits)} VALID HITS", 0, "green", None
    return title, action, timer, last, fraction, color, lit


def update_boxing(b, task):
    from pxr import Gf, UsdGeom

    from roboclock.envs.sports.arcade.refinement import bind, pbr
    from roboclock.envs.sports.arena.coach_display import PATH as screen

    # Labels and the window bar live in the coach screen group (design coordinates).
    title, action, timer, last, fraction, color, lit = boxing_display(task, b.time)
    if not hasattr(b, "screen_colors"):
        b.screen_colors = {
            name: pbr(b.appearance, "screen_" + name, rgb, roughness=1, emission=0.7)
            for name, rgb in {
                "green": [0.12, 0.8, 0.40],
                "amber": [0.9, 0.48, 0.08],
                "cyan": [0.08, 0.62, 0.85],
                "red": [0.9, 0.07, 0.03],
                "dark": [0.07, 0.10, 0.12],
            }.items()
        }
        for suffix in ("Title", "Subtitle"):
            UsdGeom.Imageable(b.stage.GetPrimAtPath(screen + "/Header/" + suffix)).MakeInvisible()
        vertical_label(
            b, screen + "/CoachBrand", "BOXING / REACTION", [0.65, 0.573, 1.69], [0.80, 0.032]
        )
    signature = (title, action, timer, last)
    if signature != getattr(b, "coach_signature", None):
        for path, text, z, height in [
            ("CoachRound", title, 1.635, 0.035),
            ("WorkflowStatus", action, 1.565, 0.068),
            ("CoachTimer", timer, 1.475, 0.058),
            ("CoachLast", last, 1.345, 0.035),
        ]:
            vertical_label(b, screen + "/" + path, text, [0.65, 0.573, z], [0.80, height])
        b.coach_signature = signature
    path = screen + "/CoachWindow"
    prim = b.stage.GetPrimAtPath(path)
    if not prim:
        shape = UsdGeom.Cube.Define(b.stage, path)
        shape.CreateSizeAttr(1)
        shape.AddTranslateOp()
        shape.AddScaleOp()
    xf = UsdGeom.Xformable(b.stage.GetPrimAtPath(path))
    translate, scale = xf.GetOrderedXformOps()
    width = 0.76 * max(0.002, min(1, fraction))
    translate.Set(Gf.Vec3d(0.65 - 0.38 + width / 2, 0.57, 1.405))
    scale.Set(Gf.Vec3f(width, 0.004, 0.013))
    bind(b, path, b.screen_colors[color])
    if lit != getattr(b, "lit_pad", "unset"):
        for side in ("left", "right"):
            bind(
                b,
                "/World/Controls/pad_" + side + "/Target",
                b.screen_colors["green" if lit == side else "dark"],
            )
        b.lit_pad = lit
