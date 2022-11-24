import numpy as np
import safeopt

"""
Define and implement the class of optimization problem.
"""


class OptimizationProblem:

    def __init__(self, config):
        self.config = config
        self.parameter_set = config['parameter_set']
        self.evaluated_points_list = []
        self.evaluated_objs_list = []
        self.evaluated_constrs_list = []
        self.problem_name = config['problem_name']
        self.eval_simu = config['eval_simu']
        self.simulator = None  # simulator used to get evaluation

        self.var_dim = config['var_dim']
        self.num_constrs = config['num_constrs']
        self.obj = config['obj']
        self.constrs_list = config['constrs_list']
        self.bounds = config['bounds']
        self.discretize_num_list = config['discretize_num_list']
        self.init_points = config['init_points']
        self.init_safe_points = config['init_safe_points']
        self.train_X = config['train_X']
        self.train_obj, self.train_constr = self.sample_point(self.train_X,
                                                              record=False)
        self.candidates = safeopt.\
            linearly_spaced_combinations(self.bounds, self.discretize_num_list)

    def get_minimum(self):
        obj_val, constr = self.sample_point(self.candidates, record=False)
        obj_val = obj_val.squeeze()
        feasible = np.array([True] * len(obj_val))
        for i in range(self.num_constrs):
            feasible = feasible & (constr[:, i] <= 0)

        minimum = np.min(obj_val[feasible])
        feasible_candidates = self.candidates[feasible, :]
        minimizer = feasible_candidates[np.argmin(obj_val[feasible]), :]
        return minimum, minimizer

    def sample_point(self, x, reset_init=False, record=True):
        if self.eval_simu:
            # evaluate objective and constraints simultaneously
            obj_val, constr_arr, simulator = self.obj(x, self.simulator)
            # inherite the simulator
            self.simulator = simulator

            obj_val = np.expand_dims(obj_val, axis=1)
            constraint_val_arr = np.expand_dims(constr_arr, axis=1)
        else:
            obj_val = self.obj(x)
            obj_val = np.expand_dims(obj_val, axis=1)
            constraint_val_list = []
            for g in self.constrs_list:
                constraint_val_list.append(g(x))
            constraint_val_arr = np.array(constraint_val_list).T

        if record:
            self.evaluated_points_list.append(x)
            self.evaluated_objs_list.append(obj_val)
            self.evaluated_constrs_list.append(constraint_val_arr)
        return obj_val, constraint_val_arr
