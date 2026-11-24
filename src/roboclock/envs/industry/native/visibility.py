"""Key objects and cues of every task, as world boxes, for the head-camera projection test.

``required`` objects must project with every checked corner at least 10 px inside the
640x480 rendered head image (``rig.inside_head_image``, ``robodojo_rig.intrinsics()``). ``partial``
objects are known to be clipped by the head camera's field of view and are listed with the
reason; the wrist cameras see them during manipulation. Boxes come from the built scenes
(static fixtures) or the reset pose (moving products); ``mode`` picks the corners checked:
``top`` (top face), ``box`` (all eight) or ``center``.
"""

KEY_OBJECTS = {
    41: dict(
        required=[
            ("part on loading pad", [[-0.03, 0.4, 0.933], [0.03, 0.46, 0.975]], "box"),
            ("plating bath", [[-0.58, 0.2, 0.67], [-0.1, 0.66, 0.93]], "top"),
            ("recipe lamps", [[0.0775, 0.7385, 1.0795], [0.2225, 0.7635, 1.1245]], "box"),
        ],
        partial=[
            ("rinse bath", [[0.1, 0.2, 0.67], [0.58, 0.66, 0.93]], "top",
             "front-right rim corner (0.58, 0.20) is 0.88 m right of the head camera at 0.2 m "
             "depth, past the 96 deg horizontal FOV; the working right arm pins the rig "
             "(both baths must stay within its 0.5 m reach)"),
        ],
    ),
    42: dict(
        required=[
            ("loading sector", [[-0.2285, 0.093, 0.858], [0.2285, 0.1759, 0.858]], "top"),
            ("rotary table", [[-0.465, 0.085, 0.83], [0.465, 1.015, 0.856]], "top"),
        ],
        partial=[
            ("pump on infeed cradle", [[-0.8611, 0.1906, 0.8549], [-0.7381, 0.3097, 0.9466]], "box",
             "left edge 0.47 m left of the camera at the image border; the recorded left-arm "
             "path is at its reach limit (a 4 cm base shift already leaves 4.6 mm IK error)"),
        ],
    ),
    43: dict(
        required=[
            ("cover source", [[-0.3248, 0.1702, 0.8255], [-0.2152, 0.3298, 0.8625]], "box"),
            ("housing in nest", [[-0.0548, 0.3202, 0.824], [0.0548, 0.4798, 0.858]], "box"),
            ("nest", [[-0.11, 0.265, 0.806], [0.11, 0.535, 0.824]], "top"),
            ("dispenser head", [[0.1915, 0.53, 0.862], [0.3015, 0.63, 1.285]], "center"),
            ("process beacon", [[0.425, 0.795, 0.97], [0.475, 0.845, 1.04]], "box"),
        ],
        partial=[],
    ),
    44: dict(
        required=[
            ("valve", [[-0.0717, 0.4246, 0.8868], [0.214, 0.5754, 1.0852]], "box"),
            ("leak outlet", [[0.432, 0.425, 0.9542], [0.448, 0.5, 0.9702]], "box"),
            ("spill tray", [[0.27, 0.18, 0.73], [0.61, 0.46, 0.78]], "top"),
            ("leak beacon", [[0.593, 0.723, 1.01], [0.647, 0.777, 1.08]], "box"),
        ],
        partial=[],
    ),
    45: dict(
        required=[
            ("stop button", [[-0.53, 0.3162, 0.99], [-0.43, 0.3965, 1.09]], "box"),
            ("fault panel", [[-0.68, 0.743, 0.945], [-0.56, 0.743, 1.025]], "box"),
        ],
        partial=[
            ("motor under test", [[-0.0551, 0.5398, 0.94], [0.2051, 0.7002, 1.1394]], "box",
             "0.8 m right of the head camera behind its guard; context only (the stop and lamp "
             "are the task cues); the working right arm pins the rig at the button"),
        ],
    ),
    46: dict(
        required=[
            ("capture lane", [[0.7824, 0.8, 0.9061], [1.2476, 1.02, 0.9731]], "top"),
            ("safety zone near edge", [[0.9, 0.786, 0.9326], [1.2, 0.804, 0.9406]], "box"),
            ("safety zone far edge", [[0.9, 1.016, 0.9326], [1.2, 1.034, 0.9406]], "box"),
            ("zone end marker", [[1.216, 0.8, 0.9101], [1.224, 1.02, 0.9117]], "box"),
            ("rework tray", [[0.735, 0.56, 0.8875], [0.965, 0.795, 0.9275]], "top"),
            ("bearing on the tray", [[0.835, 0.635, 0.9075], [0.905, 0.705, 0.9775]], "box"),
            ("arrest/recovery lamp", [[1.308, 1.168, 0.958], [1.392, 1.196, 1.042]], "box"),
        ],
        partial=[],
    ),
    47: dict(
        required=[
            ("kit 0 sources", [[0.1475, 0.48, 0.8875], [0.6125, 0.8, 0.9225]], "top"),
            ("kit 1 sources", [[-0.6125, 0.48, 0.8875], [-0.1475, 0.8, 0.9225]], "top"),
        ],
        partial=[
            ("kit 0 insert", [[0.47, 0.095, 1.0372], [0.85, 0.345, 1.0722]], "top",
             "the two 0.545 m cases sit beside the bases (x 0.39-0.93): at that depth the 96 deg "
             "horizontal FOV ends 0.55-0.66 m from the centre line and the near edge (y 0.095) is "
             "below the image; only the inner far corner is in view"),
            ("kit 1 insert", [[-0.85, 0.095, 1.0372], [-0.47, 0.345, 1.0722]], "top", "mirror of kit 0"),
        ],
    ),
    48: dict(
        required=[
            ("presented carton", [[-0.02, 0.8596, 0.5695], [0.26, 0.96, 0.7095]], "box"),
            ("A-01 upper deck", [[-0.06, 1.01, 0.75], [0.3, 1.28, 0.775]], "top"),
            ("B-01 lower deck", [[-0.19, 1.01, 0.425], [0.43, 1.27, 0.45]], "top"),
        ],
        partial=[],
    ),
    49: dict(
        required=[
            ("input tray", [[-0.32, 0.61, 0.8875], [0.32, 0.85, 0.9225]], "top"),
            ("scale indicator", [[-0.11, 0.934, 0.994], [0.11, 0.996, 1.046]], "box"),
            ("scale pan", [[-0.06, 0.42, 0.92785], [0.06, 0.52, 0.92785]], "top"),
            ("PASS left", [[-0.54, 0.52, 0.8875], [-0.36, 0.73, 0.9225]], "top"),
            ("REJECT left", [[-0.735, 0.52, 0.8875], [-0.555, 0.73, 0.9225]], "top"),
            ("PASS right", [[0.36, 0.52, 0.8875], [0.54, 0.73, 0.9225]], "top"),
            ("REJECT right", [[0.555, 0.52, 0.8875], [0.735, 0.73, 0.9225]], "top"),
        ],
        partial=[],
    ),
    50: dict(
        required=[
            ("pan feed front row", [[-0.28, 0.38, 0.8875], [0.28, 0.46, 0.9125]], "top"),
            ("pan feed back row", [[-0.28, 0.65, 0.8875], [0.28, 0.73, 0.9125]], "top"),
            ("left output tray", [[-0.61, 0.56, 0.89], [-0.39, 0.82, 0.915]], "top"),
            ("right output tray", [[0.39, 0.56, 0.89], [0.61, 0.82, 0.915]], "top"),
        ],
        partial=[
            ("left carrier with case", [[-0.62, 0.27, 0.9261], [-0.38, 0.59, 1.1014]], "top",
             "outer-front corner (x 0.62, y 0.27) is next to the base line, past the horizontal FOV; "
             "the nests stay where the pan feed rows leave room and the arms reach"),
            ("right carrier with case", [[0.38, 0.27, 0.9261], [0.62, 0.59, 1.1014]], "top", "mirror"),
        ],
    ),
}


def corners(box, mode):
    (x0, y0, z0), (x1, y1, z1) = box
    if mode == "center":
        return [[(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2]]
    zs = [z1] if mode == "top" else [z0, z1]
    return [[x, y, z] for x in (x0, x1) for y in (y0, y1) for z in zs]
