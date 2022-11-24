"""
Implement constrained Bayesian optimizer for our test.
"""
import numpy as np
import safeopt
import GPy
from scipy.stats import norm
from .base_optimizer import BaseEGO


class ConstrainedEI(BaseEGO):

    def __init__(self, opt_problem, constrained_BO_config):
        super().__init__(opt_problem, constrained_BO_config)
        # optimization problem and measurement noise
        self.num_eps = 1e-10

    def get_acquisition(self):
        obj_mean, obj_var = self.gp_obj.predict(self.parameter_set)

        obj_mean = obj_mean.squeeze()
        obj_var = obj_var.squeeze()
        constrain_mean_list = []
        constrain_var_list = []
        for i in range(self.opt_problem.num_constrs):
            mean, var = self.gp_constr_list[i].predict(self.parameter_set)

            constrain_mean_list.append(np.squeeze(mean))
            constrain_var_list.append(np.squeeze(var))
        constrain_mean_arr = np.array(constrain_mean_list).T
        constrain_var_arr = np.array(constrain_var_list).T

        # calculate Pr(g_i(x)<=0)
        prob_negtive = norm.cdf(0, constrain_mean_arr, constrain_var_arr)
        # calculate feasibility prob
        prob_feasible = np.prod(prob_negtive, axis=1)

        # calculate EI
        f_min = self.best_obj
        z = (f_min - obj_mean)/np.maximum(np.sqrt(obj_var), self.num_eps)
        EI = (f_min - obj_mean) * norm.cdf(z) + np.sqrt(obj_var) * norm.pdf(z)
        EIc = prob_feasible * EI
        return EIc

    def optimize(self):
        acq = self.get_acquisition()
        next_point_id = np.argmax(acq)
        next_point = self.parameter_set[next_point_id]
        return next_point
