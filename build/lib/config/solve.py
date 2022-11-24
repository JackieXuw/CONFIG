"""
Provide the interface to solve the problem.
"""
from solve_util import get_user_config
from optimization_problem import OptimizationProblem
from config import CONFIGOpt


class Problem:

    def __init__(self, obj, constraints, input_bounds, eval_budget, algorithm,
                 discretize_num_list=None, kernel_type=None, kernel_list=None):
        self.config = get_user_config(
            obj, constraints, input_bounds, eval_budget, algorithm,
            discretize_num_list, kernel_type, kernel_list
        )

    def solve(self, obj, constraints, input_bounds, eval_budget, algorithm,
              discretize_num_list=None, kernel_type=None, kernel_list=None):

        base_opt_config = {
            'noise_level': 0.0,
            'kernel_var': 0.1,
            'train_noise_level': 0.0,
            'problem_name': 'UserProblem'
        }

        lcb2_config = base_opt_config.copy()
        lcb2_config.update({
            'total_eval_num': eval_budget
            }
        )

        problem = OptimizationProblem(self.config)
        opt = CONFIGOpt(problem, lcb2_config)
        best_obj_list = [opt.best_obj]

        lcb2_opt_obj_list = opt.init_obj_val_list
        lcb2_opt_constr_list = opt.init_constr_val_list
        for _ in range(eval_budget):
            y_obj, constr_vals = opt.make_step()
            best_obj_list.append(opt.best_obj)
            lcb2_opt_obj_list.append(y_obj)
            lcb2_opt_constr_list.append(constr_vals)
