from roboclock.envs.hospital.tasks.throughput.deadline import DeadlineThroughput


class CartRestock(DeadlineThroughput):
    """After a code the crash cart is short of supplies: every item goes in the bin for its category before the next code.

    The countdown is the time until the next code (cycle_time); the restock is complete when the last item rests in
    its bin, t_last < cycle_time, strict. A wrong bin does not count.
    """

    task_id = "C2"
    name = "cart_restock"
    discipline = "logistics"

    def budget_text(self):
        return f"{self.spec.cycle_time:.1f} seconds (the next code)"

    def gen_instruction(self):
        return (
            "A code just ended and the crash cart is short of supplies. Put every item from the stock shelf in the "
            f"cart bin labeled with its category before the next code, in {self.spec.cycle_time:.0f} seconds."
        )

    def accepts(self, name, container):
        return self.layout.item_types[name] == self.layout.bin_types[container]
