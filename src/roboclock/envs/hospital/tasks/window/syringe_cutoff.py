"""A3: press the syringe pump's STOP key while the collected volume is inside its target range."""

from roboclock.envs.hospital.tasks.window.base import ProcessWindowTask


class SyringePumpCutoff(ProcessWindowTask):
    task_id = "A3"
    name = "syringe_pump_cutoff"
    bedside_module = "window_syringe"
    band = (4.5, 5.5)  # mL collected in the measuring vial
    unit = "mL"
    instruction = (
        "A syringe pump is delivering a bolus from a 50 mL syringe into a measuring vial. Press "
        "STOP when the collected volume is between 4.5 and 5.5 mL: too early is an insufficient "
        "dose, too late an excessive one."
    )

    def gen_instruction(self):
        width = self.window_close - self.window_open
        return f"{self.instruction} The volume stays in range for {width:g} seconds."
