"""The public card and physical support graph must describe the same arrangement."""

import numpy as np

from roboclock.envs.sports.arena.cup_patterns import icons, layout


def test_each_card_has_one_pyramid_and_one_row_on_opposite_sides():
    for card in range(4):
        targets, supports = layout(card)
        top = [n for n, p in targets.items() if p[2] > 0.8]
        assert len(top) == 1
        expected_side = "left" if card % 2 == 0 else "right"
        assert top[0].startswith(expected_side)
        assert len(supports[top[0]]) == 2
        assert all(supports[n] == ["Table"] for n in targets if n not in top)
        for side in ("left", "right"):
            diagram = icons(card, side)
            for i in range(3):
                assert (diagram[i][1] > 0) == (targets[f"{side}_{i}"][2] > 0.8)
            ordered = sorted(range(3), key=lambda i: (diagram[i][1], diagram[i][0]))
            assert ordered == ([0, 1, 2] if card % 2 == 0 else [2, 0, 1])


def test_next_card_changes_every_cup_destination_and_upper_color():
    first, _ = layout(0)
    second, supports = layout(1)
    assert all(not np.allclose(first[n], second[n]) for n in first)
    assert supports["right_1"] == ["right_2", "right_0"]
