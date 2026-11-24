"""C1 pharmacy order bins: fill every RX bin's order from the two stock risers (score_version 2).

Jobs (docs/c-class-timing.md): Q = spec.num_items = 8 ordered vials. A vial earns credit (one job)
when it rests released, still and seated in a bin whose order still needs that drug; its job time
is that moment. The two spare vials (one per riser) are distractors, not jobs.
  physical  <=>  all 8 ordered vials credited, each still released inside its bin at the end, no
                 drop, and no other vial left in a bin (final misplacements <= 0 at 95 % accuracy).
  temporal  <=>  t_Q (the 8th credit) <= T_tier = m_tier * t_o, judged at every tier from one run.
Accuracy (credited / every placement) is reported, not judged. Placements are measured from
simulator state only: the oracle cannot set object poses or declare success.
"""

from roboclock.envs.hospital.mdp.events import BLUE, GREEN, SceneCommand
from roboclock.envs.hospital.tasks.base import WardThroughputTask

BAR = "c1_shift_bar"  # countdown bar on the shift board, beside the dial
COUNT = "c1_count"  # FILLED k / N readout
ERRORS = "c1_errors"  # WRONG k readout
MAX_ERRORS_SHOWN = 3  # "ERRORS 3+" covers anything above


class PharmacyOrderBins(WardThroughputTask):
    task_id = "C1"
    name = "pharmacy_order_bins"
    discipline = "pharmacy"
    bedside_module = "ward_c1"
    instruction = (
        "Pick-to-light style order picking at a pharmacy pick face, counted per order line (one "
        "vial is one line). Each bin's RX card lists the colored vials its order needs. Pick the "
        "right vials from the medication shelf and put each into the bin whose order needs it."
    )

    def __init__(self, spec):
        super().__init__(spec)
        self.remaining = {}
        self._shown = {}

    # --- setup -------------------------------------------------------------------------------

    def bind(self, layout, placement_dwell=0.2):
        super().bind(layout, placement_dwell)
        self.remaining = {name: dict(order) for name, order in layout.orders.items()}

    def goal_text(self):
        return (
            f"Fill all {self.jobs_required} ordered vials as quickly as possible. Leave the spare "
            "vials on the shelf and put no vial in a bin whose order does not need it."
        )

    # --- scoring -----------------------------------------------------------------------------

    def accepts(self, name, container):
        kind = self.layout.item_types.get(name)
        order = self.remaining.get(container, {})
        if kind is not None and order.get(kind, 0) > 0:
            order[kind] -= 1
            return True
        return False

    def evaluate(self, state, dt):
        super().evaluate(state, dt)
        self.goal_achieved = all(v == 0 for order in self.remaining.values() for v in order.values())

    @property
    def placements(self) -> int:
        return len(self.attempted | self.dropped)

    @property
    def accuracy(self) -> float:
        return len(self.correct) / max(1, self.placements)

    # --- display -----------------------------------------------------------------------------

    def _show(self, readout, value):
        """Switch a board readout to `value` (sent through the scene's timer channel)."""
        if self._shown.get(readout) == value:
            return []
        self._shown[readout] = value
        return [SceneCommand("timer", readout, (value, "green"))]

    def process(self, dt):
        fraction = (self.spec.max_time - self.time) / self.spec.max_time
        light = GREEN if self.goal_achieved else BLUE
        commands = [SceneCommand("indicator", "indicator", light)]
        commands += self.timer("indicator", fraction)
        commands += self.timer(BAR, fraction)
        commands += self._show(COUNT, min(len(self.correct), self.spec.num_items))
        commands += self._show(ERRORS, min(self.placements - len(self.correct), MAX_ERRORS_SHOWN))
        return commands

    def job_times_list(self):
        """One entry per ordered vial, bin by bin in the order of the RX cards."""
        times = []
        for container, order in self.layout.orders.items():
            for kind, count in order.items():
                done = sorted(
                    self.job_times[name]
                    for name, where in self.counted.items()
                    if where == container and self.layout.item_types.get(name) == kind
                    and name in self.job_times
                )[:count]
                times += done + [None] * (count - len(done))
        return times

    def metrics(self):
        result = super().metrics()
        fulfilled = 0
        for container, order in self.layout.orders.items():
            fulfilled += bool(order) and all(v == 0 for v in self.remaining[container].values())
        result.update(
            correct_placements=len(self.correct),
            wrong_placements=self.placements - len(self.correct),
            attempted_operations=self.placements,
            accuracy=self.accuracy,
            required_accuracy=self.spec.required_accuracy,
            orders_fulfilled=int(fulfilled),
            orders_total=len([o for o in self.layout.orders.values() if o]),
            placement_bins=dict(self.counted),
        )
        return result
