from jaisalab.safety_constraints import BaseConstraint

class InventoryConstraints(BaseConstraint):
    def __init__(self, max_value=1, **kwargs):
        super().__init__(max_value, **kwargs)

    def evaluate(self, paths):
        """If the replenishment order is above the capacity or the inventory constraints
        for any stage of the supply chain pipeline then a cost of one is associated with that 
        stage. The costs returned will thus have shape (N*T, |Stages|-1]) i.e the number of periods per episode
        times the number of episodes in the first dimension and the shape of the pipeline as the second.
        """
        R = paths.env_infos["replenishment_quantity"]
        Im1 = paths.env_infos["inventory_constraint"]
        c =  paths.env_infos["capacity_constraint"]

        print("these are the replenishment quantities including backlogged sales:")
        print(R)

        print("These are the inventory constraints:")
        print(Im1)

        print("These are the supply constraints:")
        print(c)


