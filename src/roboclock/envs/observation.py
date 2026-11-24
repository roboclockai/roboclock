"""Observations whose camera images render only when a policy reads them.

The evaluator's time model (roboclock.evaluation.evaluator) samples an observation at
t_obs, replays the policy's inference latency and then executes its action. A chunked
policy reads one observation per query and ignores the ones produced while its chunk
plays out. Rendering those frames costs most of an episode's wall time and changes no
result, so each domain returns an `Observation`:

- The proprioception (joints, end-effector poses, gripper widths, sim_time, instruction)
  is measured when the observation is made.
- The images (head and wrist RGB, depth when enabled) are rendered the first time any of
  them is read, at the same simulation instant. Reading them after the simulation has
  advanced raises instead of returning a later scene.

`Observation` is a `dict`, so Gym spaces, `dict(obs)`, `{**obs}` and JSON see every key.
Anything that enumerates values (`items()`, `values()`, `copy()`, comparisons,
pickling) renders first.
"""

import time
from collections.abc import KeysView


class StaleObservationError(RuntimeError):
    """An observation's images were read after the simulation advanced past it."""


class Observation(dict):
    def __init__(self, state, image_keys, render, clock):
        """`render()` returns exactly `image_keys` for the current scene; `clock()` reads
        the simulation time, which must be unchanged when the images are rendered."""
        super().__init__(state)
        image_keys = tuple(image_keys)
        overlap = set(image_keys) & set(state)
        if overlap:
            raise ValueError(f"Image keys {sorted(overlap)} are also state entries")
        self._pending = image_keys
        self._image_keys = image_keys
        self._render_images = render
        self._clock = clock
        self._time = clock()
        self.render_seconds = 0.0

    @property
    def image_keys(self):
        return self._image_keys

    @property
    def rendered(self):
        """Whether the images have been rendered (always true without image keys)."""
        return not self._pending

    def render(self):
        """Render the images now if they have not been rendered; returns self."""
        if self._pending:
            now = self._clock()
            if now != self._time:
                raise StaleObservationError(
                    f"Observation from t={self._time:.4f} s read at t={now:.4f} s; "
                    "read its images before stepping the environment"
                )
            start = time.perf_counter()
            images = self._render_images()
            self.render_seconds += time.perf_counter() - start
            if set(images) != set(self._pending):
                raise RuntimeError(
                    f"Renderer returned {sorted(images)}, expected {sorted(self._pending)}"
                )
            state = dict(dict.items(self))
            dict.clear(self)
            # Images first, as the domains' eager observations listed them.
            dict.update(self, {key: images[key] for key in self._pending})
            dict.update(self, state)
            self._pending = ()
        return self

    # Reads that need an image render it; reads of state entries never do.
    def __getitem__(self, key):
        if key in self._pending:
            self.render()
        return dict.__getitem__(self, key)

    def __contains__(self, key):
        return key in self._pending or dict.__contains__(self, key)

    def __iter__(self):
        return iter([*self._pending, *dict.keys(self)])

    def __len__(self):
        return dict.__len__(self) + len(self._pending)

    def keys(self):
        return KeysView(self)

    def get(self, key, default=None):
        return self[key] if key in self else default

    def items(self):
        return dict.items(self.render())

    def values(self):
        return dict.values(self.render())

    def copy(self):
        return dict(self.items())

    def __setitem__(self, key, value):
        if key in self._pending:
            self.render()
        dict.__setitem__(self, key, value)

    def __delitem__(self, key):
        if key in self._pending:
            self.render()
        dict.__delitem__(self, key)

    def pop(self, key, *default):
        if key in self._pending:
            self.render()
        return dict.pop(self, key, *default)

    def setdefault(self, key, default=None):
        if key in self._pending:
            self.render()
        return dict.setdefault(self, key, default)

    def __eq__(self, other):
        if isinstance(other, Observation):
            other.render()
        return dict.__eq__(self.render(), other)

    def __ne__(self, other):
        return not self == other

    __hash__ = None

    def __reduce__(self):
        return dict, (self.copy(),)

    def __reduce_ex__(self, protocol):
        return self.__reduce__()

    def __repr__(self):
        if not self._pending:
            return dict.__repr__(self)
        return f"Observation({dict.__repr__(self)}, unrendered={list(self._pending)})"


def _render_first(name):
    method = getattr(dict, name)

    def wrapper(self, *args, **kwargs):
        return method(self.render(), *args, **kwargs)

    wrapper.__name__ = name
    return wrapper


# Bulk operations act on the complete dict.
for _name in ("popitem", "update", "clear", "__ior__", "__or__", "__ror__"):
    setattr(Observation, _name, _render_first(_name))


def image_keys(observation_space):
    """Keys of the camera entries (two or more dimensions) of a Dict observation space."""
    return tuple(
        key for key, space in observation_space.spaces.items() if len(space.shape or ()) >= 2
    )
