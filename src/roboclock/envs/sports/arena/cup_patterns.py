"""Asymmetric public cards, placement order and required support topology."""

import numpy as np


def layout(card):
    targets, supports = {}, {}
    for side, sign in [("left", 1), ("right", -1)]:
        pyramid = (side == "left") == (card % 2 == 0)
        y = sign * (0.17 if card % 2 == 0 else 0.34)
        order = [0, 1, 2] if card % 2 == 0 else [2, 0, 1]
        for slot, index in enumerate(order):
            name = f"{side}_{index}"
            x = (0.29 + slot * 0.065 if slot < 2 else 0.3225) if pyramid else 0.21 + slot * 0.07
            z = 0.858 if pyramid and slot == 2 else 0.786
            targets[name] = np.array([x, y, z])
            supports[name] = (
                [f"{side}_{j}" for j in order[:2]] if pyramid and slot == 2 else ["Table"]
            )
    return targets, supports


def icons(card, side):
    targets, _ = layout(card)
    pyramid = (side == "left") == (card % 2 == 0)
    ordered = sorted(range(3), key=lambda i: (targets[f"{side}_{i}"][2], targets[f"{side}_{i}"][0]))
    return {
        index: (
            (-0.022 if slot == 0 else 0.022 if slot == 1 else 0) if pyramid else (slot - 1) * 0.043,
            0.04 if pyramid and slot == 2 else 0,
        )
        for slot, index in enumerate(ordered)
    }
