import numpy as np
import safeopt


def get_fun_var(func, parameter_set, var_dim, random_sample_num_per_dim=3):
    parameter_num, _ = parameter_set.shape
    random_sample_num = random_sample_num_per_dim ** var_dim
    sample_points = parameter_set[
        np.random.choice(parameter_num, size=random_sample_num)
    ]
    func_vals = func(sample_points)
    var = np.std(func_vals) ** 2
    return var

def get_user_config(obj, constraints, input_bounds, eval_budget, algorithm,
          discretize_num_list=None, kernel_type=None, kernel_list=None):
    config = dict()
    config['problem_name'] = 'UserProblem'

    problem_dim = len(input_bounds)

    if kernel_type is None:
        gp_kernel = 'Gaussian'
    else:
        gp_kernel = 'kernel_type'

    config['var_dim'] = problem_dim
    if discretize_num_list is None:
        # have 100 grids in each dimension by default
        config['discretize_num_list'] = [100] * problem_dim
    else:
        config['discretize_num_list'] = discretize_num_list

    config['num_constrs'] = len(constraints)

    config['bounds'] = input_bounds
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
            config['discretize_num_list']
        )

    if kernel_list is not None:
        config['kernel'] = [kernel, kernel.copy()]
    else:
        # define kernel
        kernel_list = []
        obv_lengthscales = [
            (config['bounds'][k][1] - config['bounds'][k][0])/4.0
            for k in range(len(config['bounds']))
        ]
        if gp_kernel == 'Gaussian':
            obj_var = get_fun_var(
                config['obj'], parameter_set, config['var_dim']
            )
            obj_kernel = GPy.kern.RBF(
                input_dim=len(config['bounds']), variance=obj_var,
                lengthscale=obv_lengthscales, ARD=True)
            kernel_list.append(obj_kernel)
            for j in range(config['num_constrs']):
                constr_var = get_fun_var(
                    constraints[j], parameter_set, config['var_dim']
                )
                kernel_list.append(
                    GPy.kern.RBF(input_dim=len(config['bounds']),
                                 variance=constr_var,
                    lengthscale=obv_lengthscales, ARD=True
                )
                )


    # Initial safe point
    x0 = np.expand_dims(np.array(parameter_set[0]), axis=0)
    # np.zeros((1, len(config['bounds'])))

    safe_margin = 0.2
    func = sample_safe_fun(
            kernel, config, noise_var, gp_kernel, safe_margin)
    func_min = np.min(func(parameter_set))
    config['f_min'] = 0.0  # func_min


    config['obj'] = obj  #f
    config['constrs_list'] = constraints   # [g_1]
    config['init_safe_points'] = x0
    return config
