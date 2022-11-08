import numpy as np
import math
import GPy
import safeopt

"""
Define some utility functions for the test of safe Bayesian optimization,
constrained Bayesian optimization, and our method.
"""


# Generate function with safe initial point at x=0
def sample_safe_fun(kernel, config, noise_var, gp_kernel, safe_margin):
    while True:
        fun = safeopt.sample_gp_function(kernel, config['bounds'],
                                         noise_var, 100)
        if fun(0, noise=False) < -safe_margin:
            break
    return fun


def get_sinusodal_config(config):
    config['var_dim'] = 2
    config['discretize_num_list'] = [100 for _ in range(config['var_dim'])]
    config['num_constrs'] = 1
    config['bounds'] = [(0, 6), (0, 6)]
    config['train_X'] = safeopt.linearly_spaced_combinations(
            config['bounds'],
            [20 for _ in range(config['var_dim'])]
        )

    def f(x_f):
        return np.cos(2 * x_f[:, 0]) * np.cos(x_f[:, 1]) + np.sin(x_f[:, 0])

    def g_1(x_g1):
        return np.cos(x_g1[:, 0]) * np.cos(x_g1[:, 1]) - \
                np.sin(x_g1[:, 0]) * np.sin(x_g1[:, 1]) + 0.2

    config['obj'] = f
    config['constrs_list'] = [g_1]
    config['init_safe_points'] = np.array([[math.pi * 0.5, math.pi * 0.5]])


def get_GP_sample_single_func(config, problem_name, problem_dim=None,
                              gp_kernel=None, init_points_id=0):
    if problem_dim is None:
        problem_dim = 1
    if gp_kernel is None:
        gp_kernel = 'Gaussian'
    config['var_dim'] = problem_dim
    config['discretize_num_list'] = [100] * problem_dim
    config['num_constrs'] = 1
    config['bounds'] = [(-10, 10)] * problem_dim
    config['train_X'] = safeopt.linearly_spaced_combinations(
        config['bounds'],
        config['discretize_num_list']
    )

    # Measurement noise
    noise_var = 0.00

    # Bounds on the inputs variable
    parameter_set = \
        safeopt.linearly_spaced_combinations(
            config['bounds'],
            config['discretize_num_list'])

    # Define Kernel
    if gp_kernel == 'Gaussian':
        kernel = GPy.kern.RBF(input_dim=len(config['bounds']), variance=2.,
                              lengthscale=1.0, ARD=True)
    if gp_kernel == 'poly':
        kernel = GPy.kern.Poly(input_dim=len(config['bounds']),
                               variance=2.0,
                               scale=1.0,
                               order=1)

    # Initial safe point
    x0 = np.zeros((1, len(config['bounds'])))

    safe_margin = 0.2
    func = sample_safe_fun(
            kernel, config, noise_var, gp_kernel, safe_margin)
    func_min = np.min(func(parameter_set))
    config['f_min'] = func_min

    def f(x):
        return func(x, noise=False).squeeze(axis=1)

    def g_1(x):
        return func(x, noise=False).squeeze(axis=1)

    config['obj'] = f
    config['constrs_list'] = [g_1]
    config['init_safe_points'] = x0
    config['kernel'] = [kernel, kernel.copy()]
    return config


def get_config(problem_name, problem_dim=None, gp_kernel=None,
               init_points_id=0):
    """
    Input: problem_name
    Output: configuration of the constrained problem, including variable
    dimension, number of constraints, objective function and constraint
    function.
    """
    config = dict()
    config['problem_name'] = problem_name

    if problem_name == 'sinusodal':
        config = get_sinusodal_config(config)

    if problem_name == 'GP_sample_single_func':
        config = get_GP_sample_single_func(
            config, problem_name, problem_dim, gp_kernel,
            init_points_id)

    if problem_name == 'GP_sample_two_funcs':
        if problem_dim is None:
            problem_dim = 1
        if gp_kernel is None:
            gp_kernel = 'Gaussian'
        config['var_dim'] = problem_dim
        config['discretize_num_list'] = [100] * problem_dim
        config['num_constrs'] = 1
        config['bounds'] = [(-10, 10)] * problem_dim
        config['train_X'] = safeopt.linearly_spaced_combinations(
            config['bounds'],
            config['discretize_num_list']
        )
        config['parameter_set'] = config['train_X']
        config['eval_simu'] = False
        # Measurement noise
        noise_var = 0.00  # 0.05 ** 2

        # Bounds on the inputs variable
        parameter_set = \
            safeopt.linearly_spaced_combinations(config['bounds'],
                                                 config['discretize_num_list'])

        # Define Kernel
        if gp_kernel == 'Gaussian':
            kernel = GPy.kern.RBF(input_dim=len(config['bounds']), variance=2.,
                                  lengthscale=1.0, ARD=True)
        if gp_kernel == 'poly':
            kernel = GPy.kern.Poly(input_dim=len(config['bounds']),
                                   variance=2.0,
                                   scale=1.0,
                                   order=1)

        # Initial safe point
        x0 = np.zeros((1, len(config['bounds'])))

        constr_func = sample_safe_fun(
            kernel, config, noise_var, gp_kernel, safe_margin=0.2)
        obj_func = safeopt.sample_gp_function(kernel, config['bounds'],
                                              noise_var, 100)
        func_feasible_min = np.min(
            obj_func(parameter_set)[constr_func(parameter_set) <= 0])
        config['f_min'] = func_feasible_min

        def f(x):
            return obj_func(x, noise=False).squeeze(axis=1)

        def g_1(x):
            return constr_func(x, noise=False).squeeze(axis=1)

        config['obj'] = f
        config['constrs_list'] = [g_1]
        config['init_safe_points'] = x0
        config['kernel'] = [kernel, kernel.copy()]
        config['init_points'] = x0
    return config


if __name__ == '__main__':
    a = get_config('GP_sample_two_funcs')
    print(a)
