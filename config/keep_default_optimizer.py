"""
Implement keep default optimizer.
"""
import numpy as np
import safeopt
from .base_optimizer import BaseEGO


class KeepDefaultOpt(BaseEGO):

    def __init__(self, opt_problem, keep_default_config):
        super().__init__(opt_problem, keep_default_config)
        self.x_default = keep_default_config['default_x']

    def optimize(self, evalute_point=None):
        if evalute_point is None:
            evalute_point = self.x_default
        return evalute_point
