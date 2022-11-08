"""
Implement violation-aware Bayesian optimizer.
"""
import numpy as np
import safeopt
import GPy
from scipy.stats import norm


class ViolationAwareBO:

    def __init__(self, opt_problem, violation_aware_BO_config):
        # optimization problem and measurement noise
        if 'normalize_input' in violation_aware_BO_config.keys():
            self.normalize_input = violation_aware_BO_config['normalize_input']
        else:
            self.normalize_input = True

        # Pr(cost <= beta * budget) >= 1 - \epsilon
        if 'beta_func' in violation_aware_BO_config.keys():
            self.beta_func = violation_aware_BO_config['beta_func']
        else:
            self.beta_func = lambda t: 1

        self.num_eps = 1e-10   # epsilon for numerical value
        self.opt_problem = opt_problem
        self.total_vio_budgets = violation_aware_BO_config['total_vio_budgets']
        self.noise_level = violation_aware_BO_config['noise_level']
        if 'train_noise_level' in violation_aware_BO_config.keys():
            self.train_noise_level = violation_aware_BO_config[
                'train_noise_level']
        else:
            self.train_noise_level = 10.0
        self.kernel_var = violation_aware_BO_config['kernel_var']
        self.prob_eps = violation_aware_BO_config['prob_eps']
        self.beta_0 = violation_aware_BO_config['beta_0']
        self.total_eval_num = violation_aware_BO_config['total_eval_num']

        # Bounds on the inputs variable
        self.bounds = opt_problem.bounds
        self.discret_num_list = opt_problem.discretize_num_list

        if 'kernel_type' in violation_aware_BO_config.keys():
            self.set_kernel(kernel_type=violation_aware_BO_config[
                'kernel_type'])
        else:
            self.set_kernel()

        # set of parameters
        self.parameter_set = safeopt.linearly_spaced_combinations(
            self.bounds,
            self.discret_num_list
        )

        # Initial safe point
        self.x0_arr = opt_problem.init_safe_points
        self.query_points_list = []
        self.query_point_obj = []
        self.query_point_constrs = []
        self.S = []
        # self.kernel_list = []
        init_obj_val_arr, init_constr_val_arr = \
            self.get_obj_constr_val(self.x0_arr)
        self.init_obj_val_arr = init_obj_val_arr
        self.init_constr_val_arr = init_constr_val_arr
        self.best_obj = np.min(init_obj_val_arr)
        self.gp_obj_mean = 0
        self.setup_optimizer()

    def get_kernel_train_noise_level(self, noise_fraction=1.0/3.0):
        obj_max = np.max(self.opt_problem.train_obj)
        obj_min = np.min(self.opt_problem.train_obj)
        obj_range = obj_max - obj_min
        obj_noise_level = obj_range * noise_fraction
        constr_noise_level_list = []
        for i in range(self.opt_problem.num_constrs):
            constr_obj = np.expand_dims(self.opt_problem.train_constr[:, i],
                                        axis=1)
            constr_max = np.max(constr_obj)
            constr_min = np.min(constr_obj)
            constr_range = constr_max - constr_min
            constr_noise_level = constr_range * noise_fraction
            constr_noise_level_list.append(constr_noise_level)
        return obj_noise_level, constr_noise_level_list

    def set_kernel(self, kernel_type='Gaussian', noise_fraction=1.0 / 2.0):
        if 'kernel' in self.opt_problem.config.keys():
            # print('kernel in opt problem config.')
            self.kernel_list = self.opt_problem.config['kernel']
            return 0
        obj_noise_level, constr_noise_level_list = \
            self.get_kernel_train_noise_level(noise_fraction)
        kernel_list = []
        opt_problem = self.opt_problem

        if kernel_type == 'Gaussian':
            kernel_list.append(GPy.kern.RBF(input_dim=len(self.bounds),
                                            variance=self.kernel_var,
                                            lengthscale=5.0,
                                            ARD=True
                                            )
                               )
            for i in range(opt_problem.num_constrs):
                kernel_list.append(GPy.kern.RBF(input_dim=len(self.bounds),
                                                variance=self.kernel_var,
                                                lengthscale=5.0,
                                                ARD=True
                                                )
                                   )
        elif kernel_type == 'polynomial':
            kernel_list.append(GPy.kern.Poly(input_dim=len(self.bounds),
                                             variance=self.kernel_var,
                                             scale=5.0,
                                             order=4
                                             )
                               )
            for i in range(opt_problem.num_constrs):
                kernel_list.append(GPy.kern.Poly(input_dim=len(self.bounds),
                                                 variance=self.kernel_var,
                                                 scale=5.0,
                                                 order=4
                                                 )
                                   )
        else:
            raise Exception('Unknown kernel type!')

        num_train_data, _ = opt_problem.train_obj.shape
        obj_noise = obj_noise_level * np.random.randn(
            num_train_data, 1)
        obj_gp = GPy.models.GPRegression(
            opt_problem.train_X,
            opt_problem.train_obj + obj_noise,
            self.kernel_list[0]
        )
        obj_gp.optimize()

        for i in range(opt_problem.num_constrs):
            constr_obj = np.expand_dims(opt_problem.train_constr[:, i],
                                        axis=1)
            constr_noise = constr_noise_level_list[i] * np.random.randn(
                num_train_data, 1)
            constr_gp = GPy.models.GPRegression(
                opt_problem.train_X,
                constr_obj + constr_noise,
                kernel_list[i+1])
            constr_gp.optimize()

        self.kernel_list = kernel_list

    def setup_optimizer(self):
        # The statistical model of our objective function and safety constraint

        self.gp_obj = GPy.models.GPRegression(self.x0_arr,
                                              self.init_obj_val_arr,
                                              self.kernel_list[0],
                                              noise_var=self.noise_level ** 2)
        self.gp_constr_list = []
        self.gp_constr_mean_list = []
        for i in range(self.opt_problem.num_constrs):
            if self.normalize_input:
                gp_constr_mean = np.mean(self.init_constr_val_arr[:, i])
            else:
                gp_constr_mean = 0
            self.gp_constr_list.append(
                GPy.models.GPRegression(self.x0_arr,
                                        np.expand_dims(
                                            self.init_constr_val_arr[:, i],
                                            axis=1),
                                        self.kernel_list[i+1],
                                        noise_var=self.noise_level ** 2))
            self.gp_constr_mean_list.append(gp_constr_mean)

        self.opt = safeopt.SafeOpt([self.gp_obj] + self.gp_constr_list,
                                   self.parameter_set,
                                   [-np.inf] + [0.] *
                                   self.opt_problem.num_constrs,
                                   lipschitz=None,
                                   threshold=0.1
                                   )
        self.curr_budgets = self.total_vio_budgets
        self.curr_eval_budget = self.total_eval_num
        self.cumu_vio_cost = np.zeros(self.opt_problem.num_constrs)

    def get_obj_constr_val(self, x_arr, noise=False):
        obj_val_arr, constr_val_arr = self.opt_problem.sample_point(x_arr)
        return obj_val_arr, constr_val_arr

    def plot(self):
        # Plot the GP
        self.opt.plot(100)
        # Plot the true function
        y, constr_val = self.get_obj_constr_val(self.parameter_set,
                                                noise=False)

    def get_acquisition(self,
                        type='budget_aware_EIc', gp_package='gpy', lcb_beta=3):
        obj_mean, obj_var = self.gp_obj.predict(self.parameter_set)

        obj_mean = obj_mean + self.gp_obj_mean
        obj_mean = obj_mean.squeeze()
        obj_var = obj_var.squeeze()
        constrain_mean_list = []
        constrain_var_list = []
        for i in range(self.opt_problem.num_constrs):
            mean, var = self.gp_constr_list[i].predict(self.parameter_set)

            mean = mean + self.gp_constr_mean_list[i]
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
        lcb = obj_mean + lcb_beta * np.maximum(np.sqrt(obj_var), self.num_eps)

        # calculate Pr(c_i([g_i(x)]^+)<=B_{i,t}/beta_t)
        curr_beta = self.get_beta()
        curr_cost_allocated = self.curr_budgets/curr_beta
        allowed_vio = self.opt_problem.get_vio_from_cost(curr_cost_allocated)
        prob_not_use_up_budget = norm.cdf(allowed_vio, constrain_mean_arr,
                                          constrain_var_arr)
        prob_all_not_use_up_budget = np.prod(prob_not_use_up_budget, axis=1)

        self.S = self.parameter_set[
            (prob_all_not_use_up_budget >= 1 - self.prob_eps)]
        if type == 'constrained_EI':
            return EIc
        if type == 'budget_aware_EIc':
            EIc_indicated = EIc * (prob_all_not_use_up_budget >=
                                   1 - self.prob_eps)
            return EIc_indicated
        if type == 'budget_aware_EI':
            EI_indicated = EI * (prob_all_not_use_up_budget >=
                                 1 - self.prob_eps)
            return EI_indicated
        if type == 'budget_aware_lcb':
            return lcb, prob_all_not_use_up_budget >= 1 - self.prob_eps

    def get_beta(self):
        return min(
            max(self.curr_eval_budget, 1),
            self.beta_func(self.curr_eval_budget)
        )

    def optimize(self, type='budget_aware_EI', gp_package='gpy'):
        if type == 'budget_aware_EI':
            acq = self.get_acquisition(gp_package=gp_package)
        assert np.any(acq > 0)
        next_point_id = np.argmax(acq)
        next_point = self.parameter_set[next_point_id]
        return next_point

    def make_step(self, update_gp=False, gp_package='gpy'):
        if np.any(self.curr_budgets < 0) or self.curr_eval_budget <= 0:
            return None, None
        x_next = self.optimize(gp_package=gp_package)
        x_next = np.array([x_next])
        # Get a measurement from the real system
        y_obj, constr_vals = self.get_obj_constr_val(x_next)

        self.query_points_list.append(x_next)
        self.query_point_obj.append(y_obj)
        self.query_point_constrs.append(constr_vals)

        vio_cost = self.opt_problem.get_total_violation_cost(constr_vals)
        vio_cost = np.squeeze(vio_cost)
        self.curr_budgets -= vio_cost
        if np.all(constr_vals <= 0) and np.all(self.curr_budgets >= 0):
            # update best objective if we get a feasible point
            self.best_obj = np.min([y_obj[0, 0], self.best_obj])
        y_meas = np.hstack((y_obj, constr_vals))
        violation_cost = self.opt_problem.get_total_violation_cost(constr_vals)
        violation_total_cost = np.sum(violation_cost, axis=0)
        self.cumu_vio_cost = self.cumu_vio_cost + violation_total_cost

        # Add this to the GP model
        if self.normalize_input:
            prev_X = self.opt.gps[0].X
            prev_obj = self.opt.gps[0].Y + self.gp_obj_mean
            prev_constr_list = []
            for i in range(self.opt_problem.num_constrs):
                prev_constr_list.append(self.opt.gps[i+1].Y)
            new_X = np.vstack([prev_X, x_next])
            new_obj = np.vstack([prev_obj, y_obj])
            self.gp_obj_mean = np.mean(new_obj)
            new_obj = new_obj - self.gp_obj_mean
            self.opt.gps[0].set_XY(new_X, new_obj)
            new_constrs_list = []
            for i in range(self.opt_problem.num_constrs):
                new_constr = np.vstack(
                    [prev_constr_list[i],
                     np.expand_dims(constr_vals[:, i], axis=1)])
                self.gp_constr_mean_list[i] = np.mean(new_constr)
                new_constr = new_constr - self.gp_constr_mean_list[i]
                new_constrs_list.append(new_constr)
                self.opt.gps[i+1].set_XY(new_X, new_constr)

            self.gps_torch.set_new_datas(
                new_X, new_obj, new_constrs_list, update_hyper_params=True)
        else:
            self.opt.add_new_data_point(x_next, y_meas)
        self.curr_eval_budget -= 1
        return y_obj, constr_vals
