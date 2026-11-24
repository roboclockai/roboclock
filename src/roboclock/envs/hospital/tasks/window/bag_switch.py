"""A2: turn the selector to the second bag after bag 1 empties, before its drip chamber runs dry.

Described as what it is at this time scale: two small (100 mL) bags run wide open by gravity on a
training line, so once bag 1 is empty its drip chamber runs dry within seconds (window 3-8 s).
"""

from roboclock.envs.hospital.tasks.window.base import ProcessWindowTask


class SequentialBagSwitch(ProcessWindowTask):
    task_id = "A2"
    name = "sequential_bag_switch"
    bedside_module = "window_bags"
    band = (95.0, 100.0)  # percent of bag 1 delivered: END mark .. drip chamber dry
    unit = "% bag 1"
    instruction = (
        "Two color-coded 100 mL saline bags, BAG 1 (blue) and BAG 2 (amber), run wide open by gravity "
        "into a training line through a selector valve. Once bag 1 is empty its drip chamber runs dry "
        "within seconds. Turn the selector's black pointer a quarter turn from the BAG 1 port to the "
        "BAG 2 port after bag 1's fluid reaches its green END mark and before its drip chamber runs "
        "dry: too early interrupts bag 1, too late lets air into the line."
    )

    def gen_instruction(self):
        width = self.window_close - self.window_open
        return f"{self.instruction} The switch window lasts {width:g} seconds."
