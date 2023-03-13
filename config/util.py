import numpy as np
import math
import GPy
import safeopt

"""
Define some utility functions for the test of safe Bayesian optimization, EPBO,
constrained Bayesian optimization, and our method.
"""

def get_ApartTherm_kpis(controller, params):
    pass
    # [TODO] Implement the get_ApartTherm_kpis functions
    return None, None

cost_funcs = {
        'square': lambda x: np.square(x),
        'exp': lambda x: np.exp(x) - 1,
        'linear': lambda x: x
    }
cost_funcs_inv = {
        'square': lambda x: np.sqrt(x),
        'exp': lambda x: np.log(x+1),
        'linear': lambda x: x
    }

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
    return config


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
    config['parameter_set'] = parameter_set
    config['eval_simu'] = False

    def f(x):
        return func(x, noise=False).squeeze(axis=1)

    def g_1(x):
        return func(x, noise=False).squeeze(axis=1)

    config['obj'] = f
    config['constrs_list'] = [g_1]
    config['init_safe_points'] = x0
    config['kernel'] = [kernel, kernel.copy()]
    config['init_points'] = x0
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
    if problem_name == 'energym_apartment_therm_tune':
        if problem_dim is None:
            problem_dim = 2
        if gp_kernel is None:
            gp_kernel = 'Gaussian'
        config['var_dim'] = problem_dim
        config['discretize_num_list'] = [100] * problem_dim
        config['num_constrs'] = 1
        config['bounds'] = [(0.01, 0.3), (15, 20)] #[(0.05, 0.95), (18, 25)] #[(0.05, 0.95), (0.05, 0.95)] # [(0.05, 0.45), (0.5, 0.95)] #[(-10, 10)] * problem_dim
        config['train_X'] = safeopt.linearly_spaced_combinations(
            config['bounds'],
            2 #config['discretize_num_list']
        )

        config['eval_simu'] = True
        config['eta_func'] = lambda t: 3/np.sqrt(t * 1.0)
        # Measurement noise
        noise_var = 0.00  # 0.05 ** 2

        # Bounds on the inputs variable
        parameter_set = \
            safeopt.linearly_spaced_combinations(config['bounds'],
                                                 config['discretize_num_list'])

        # Define Kernel
        if gp_kernel == 'Gaussian':
            kernel = GPy.kern.RBF(input_dim=len(config['bounds']), variance=1.,
                                  lengthscale=[0.3/3.0, 2.0/3.0], ARD=True)
            constr_kernel = GPy.kern.RBF(
                input_dim=len(config['bounds']), variance=1.,
                lengthscale=[0.3/3.0, 2.0/3.0], ARD=True)

        if gp_kernel == 'poly':
            kernel = GPy.kern.Poly(input_dim=len(config['bounds']),
                                   variance=2.0,
                                   scale=1.0,
                                   order=1)

        # Initial safe point
        x0 = np.array([[0.3, 20]])

        def f(x, simulator_to_use=None):
            energy_mean = 1117.9042996350206 #1500.954861111111  # 385.3506855413606
            energy_std = 112.68479944833072 # 12.634541238981747 # 15.832128159177238
            dev_mean =  1.2810264538593588 * 3.0 #0.31793574850163836 # 0.48119964518630765
            dev_std =  0.7189539259567139 #0.03637075568519829 # 0.016940298884339722
            size_batch, _ = x.shape
            energy_list = []
            dev_list = []
            #print(f'Size batch {size_batch}!')
            for k in range(size_batch):
                energy, dev = get_ApartTherm_kpis(
                    controller='P', params=(x[k, 0], x[k, 1]))
                energy_list.append([energy])
                dev_list.append([dev])
            energy_arr = (np.array(energy_list) - energy_mean) / energy_std
            dev_arr = (np.array(dev_list) - dev_mean) / dev_std
            return energy_arr, dev_arr, simulator_to_use

        def g_1(x, simulator_to_use=None):
            energy_mean = 1117.9042996350206 #1500.954861111111  # 385.3506855413606
            energy_std = 112.68479944833072 # 12.634541238981747 # 15.832128159177238
            dev_mean =  1.2810264538593588 #0.31793574850163836 # 0.48119964518630765
            dev_std =  0.7189539259567139 #0.03637075568519829 # 0.016940298884339722
            size_batch, _ = x.shape
            energy_list = []
            dev_list = []
            for k in range(size_batch):
                energy, dev = get_ApartTherm_kpis(
                    controller='P', params=(x[k, 0], x[k, 1]))
                energy_list.append([energy])
                dev_list.append([dev])
            energy_arr = (np.array(energy_list) - energy_mean) / energy_std
            dev_arr = (np.array(dev_list) - dev_mean) / dev_std
            return dev_arr, energy_arr, simulator_to_use

        config['obj'] = f
        config['constrs_list'] = [g_1]
        config['vio_cost_funcs_list'] = [cost_funcs['square']]
        config['vio_cost_funcs_inv_list'] = [cost_funcs_inv['square']]
        config['init_safe_points'] = x0
        config['kernel'] = [kernel, constr_kernel]
        print(config)


    return config


if __name__ == '__main__':
    a = get_config('GP_sample_two_funcs')
    print(a)
