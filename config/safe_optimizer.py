"""
Implement safe Bayesian optimizer for our test.
"""
import numpy as np
import safeopt
import GPy
from .base_optimizer import BaseEGO


class SafeBO(BaseEGO):

    def __init__(self, opt_problem, safe_BO_config):
        # optimization problem and measurement noise
        super().__init__(opt_problem, safe_BO_config)

        # Initial safe point
        self.x0_arr = opt_problem.init_safe_points
        if 'safe_thr' in safe_BO_config.keys():
            self.safe_thr = safe_BO_config['safe_thr']
        else:
            self.safe_thr = 0.1
        self.setup_optimizer()

    def setup_optimizer(self):
        # The statistical model of our objective function and safety constraint
        init_obj_val_arr, init_constr_val_arr = \
            self.get_obj_constr_val(self.x0_arr)
        self.init_obj_val_arr = init_obj_val_arr
        self.init_constr_val_arr = init_constr_val_arr
        self.best_obj = np.max(init_obj_val_arr[:, 0])

        self.gp_obj = GPy.models.GPRegression(
            self.x0_arr,
            init_obj_val_arr,
            self.kernel_list[0],
            noise_var=self.noise_level[0] ** 2
        )

        self.gp_constr_list = []
        for i in range(self.opt_problem.num_constrs):
            self.gp_constr_list.append(
                GPy.models.GPRegression(self.x0_arr,
                                        np.expand_dims(
                                            init_constr_val_arr[:, i], axis=1
                                        ),
                                        self.kernel_list[i+1],
                                        noise_var=self.noise_level[i+1] ** 2
                                        )
            )

        self.opt = safeopt.SafeOpt([self.gp_obj] + self.gp_constr_list,
                                   self.parameter_set,
                                   [-np.inf] + [0.] *
                                   self.opt_problem.num_constrs,
                                   lipschitz=None,
                                   threshold=self.safe_thr
                                   )

    def get_obj_constr_val(self, x_arr, noise=False):
        obj_val_arr, constr_val_arr = self.opt_problem.sample_point(x_arr)
        obj_val_arr = -1 * obj_val_arr
        constr_val_arr = -1 * constr_val_arr
        return obj_val_arr, constr_val_arr

    def make_step(self):
        x_next = self.opt.optimize()
        x_next = np.array([x_next])
        # Get a measurement from the real system
        y_obj, constr_vals = self.get_obj_constr_val(x_next)
        self.best_obj = max(self.best_obj, y_obj[0, 0])
        y_meas = np.hstack((y_obj, constr_vals))
        # Add this to the GP model
        self.opt.add_new_data_point(x_next, y_meas)
        return y_obj, constr_vals
