from dowel import logger

class BaseConstraint(object):

    def __init__(self, max_value=1., **kwargs):
        self.max_value = max_value

    def evaluate(self, paths):
        raise NotImplementedError

    def get_safety_step(self):
        return self.max_value