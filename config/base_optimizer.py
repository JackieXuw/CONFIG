"""
Implement optimizer base class.
"""
import numpy as np
import GPy


class BaseEGO:

    def __init__(self, opt_problem, base_config):
        """
        Base class for efficient global optimization.
        """
        self.inf = 1e12
        self.opt_problem = opt_problem
        self.parameter_set = opt_problem.parameter_set
        self.var_dim = opt_problem.config['var_dim']
        noise_level = base_config['noise_level']
        if type(noise_level) in [list, np.ndarray]:
            self.noise_level = noise_level
        else:
            self.noise_level = [noise_level] * (opt_problem.num_constrs + 1)
        self.kernel_list = self.opt_problem.config['kernel']

        self.x0_arr = opt_problem.init_points

        init_obj_val_arr, init_constr_val_arr = \
            self.get_obj_constr_val(self.x0_arr)

        if np.any(np.prod(init_constr_val_arr <= 0, axis=1)):
            self.best_obj = np.min(init_obj_val_arr)
            best_obj_id = np.argmin(init_obj_val_arr)
            self.best_x = self.x0_arr[best_obj_id]
        else:
            self.best_obj = self.inf
            self.best_x = None

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
                                            init_constr_val_arr[:, i], axis=1),
                                        self.kernel_list[i+1],
                                        noise_var=self.noise_level[i+1] ** 2)
            )

    def get_obj_constr_val(self, x_arr, noise=False):
        obj_val_arr, constr_val_arr = self.opt_problem.sample_point(x_arr)
        return obj_val_arr, constr_val_arr

    def get_acquisition(self):
        raise NotImplementedError

    def optimize(self):
        raise NotImplementedError

    def make_step(self):
        x_next = np.array([self.optimize()])

        # Get a measurement of the black-box function
        y_obj, constr_vals = self.get_obj_constr_val(x_next)
        if np.all(constr_vals <= 0):
            # update best objective if we get a feasible point
            if y_obj[0, 0] < self.best_obj:
                self.best_x = x_next[0]
            self.best_obj = np.min([y_obj[0, 0], self.best_obj])

        # Add this to the GP model
        prev_X = self.gp_obj.X
        prev_obj = self.gp_obj.Y
        prev_constr_list = []
        for i in range(self.opt_problem.num_constrs):
            prev_constr_list.append(self.gp_constr_list[i].Y)
        new_X = np.vstack([prev_X, x_next])
        new_obj = np.vstack([prev_obj, y_obj])
        self.gp_obj.set_XY(new_X, new_obj)
        for i in range(self.opt_problem.num_constrs):
            new_constr = np.vstack([prev_constr_list[i],
                                    np.expand_dims(constr_vals[:, i], axis=1)])
            self.gp_constr_list[i].set_XY(new_X, new_constr)
        return y_obj, constr_vals
