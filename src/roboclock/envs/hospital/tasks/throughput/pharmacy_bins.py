from roboclock.envs.hospital.tasks.throughput.deadline import DeadlineThroughput


class PharmacyBins(DeadlineThroughput):
    """A STAT pharmacy order: each bin takes exactly the items its RX label lists before the courier leaves.

    The countdown is the time until the courier leaves (cycle_time). A placement is correct while the bin's order still
    needs that item type; the order is complete when the last item is in, t_last < cycle_time, strict.
    """

    task_id = "C1"
    name = "pharmacy_bins"
    discipline = "pharmacy"

    def budget_text(self):
        return f"{self.spec.cycle_time:.1f} seconds (the courier leaves)"

    def gen_instruction(self):
        return (
            "A STAT pharmacy order is waiting for the courier. Fill each order bin from the medication shelf with "
            f"exactly the items its RX label lists, in {self.spec.cycle_time:.0f} seconds."
        )

    def bind(self, layout, placement_dwell=0.2):
        super().bind(layout, placement_dwell)
        self.remaining = {name: dict(order) for name, order in layout.orders.items()}

    def accepts(self, name, container):
        kind = self.layout.item_types[name]
        if self.remaining[container].get(kind, 0) > 0:
            self.remaining[container][kind] -= 1
            return True
        return False

    def metrics(self):
        result = super().metrics()
        fulfilled = 0
        if self.previous is not None:
            for container, order in self.layout.orders.items():
                held = {}
                for name, where in self.counted.items():
                    if where == container:
                        kind = self.layout.item_types[name]
                        held[kind] = held.get(kind, 0) + 1
                fulfilled += held == {k: v for k, v in order.items() if v}
        result["orders_fulfilled"] = fulfilled
        result["orders_total"] = len([o for o in self.layout.orders.values() if o])
        return result
