from jaisalab.safety_constraints import BaseConstraint
import numpy as np

class InventoryConstraints(BaseConstraint):
    def __init__(self, max_value=1, **kwargs):
        super().__init__(max_value, **kwargs)

    def evaluate(self, paths):
        """If the replenishment order is above the capacity or the inventory constraints
        for any stage of the supply chain pipeline then a cost of one is associated with that 
        stage. The costs returned will thus have shape (N*T, |Stages|-1]) i.e the number of periods per episode
        times the number of episodes in the first dimension and the shape of the pipeline as the second.
        """
        desired_shape =  len(paths.rewards)
        costs = np.zeros(desired_shape)
        R = paths.env_infos["replenishment_quantity"]
        Im1 = paths.env_infos["inventory_constraint"]
        c =  paths.env_infos["capacity_constraint"]

        for i, (orders, inventories, capacities) in enumerate(zip(R,Im1,c)):
            for stage_order, stage_inventory, stage_capacity in zip(orders, inventories, capacities):
                if stage_order > stage_inventory: 
                    costs[i] += 1
                if stage_order > stage_capacity:
                    costs[i] += 1
        return costs

