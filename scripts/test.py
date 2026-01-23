import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import numpy as np
from src.visymre.architectures.model import Model

from src.visymre.dclasses import BFGSParams, FitParams, NNEquation
from src.visymre.utils import load_metadata_hdf5
from functools import partial
import hydra,math,re
from sympy import symbols, exp, lambdify,sympify
from numpy import nan
from sklearn.metrics import r2_score
import ast
import warnings
import omegaconf
from typing import List
import  torch
warnings.filterwarnings("ignore")
import sympy as sp
import torch.nn.functional as F

def round_if_needed(val):
    """
    对一个 sympy 的数字进行四舍五入：保留一位小数，
    如果小数部分为 0，则返回整数类型。
    """
    num = float(val)
    rounded = round(num, 1)
    if abs(rounded - int(rounded)) < 1e-10:
        return sp.Integer(int(rounded))
    else:
        return sp.Float(rounded)

def process_expr(expr, in_exponent=False):
    """
    递归遍历 sympy 表达式 expr，寻找数字常数，
    如果不在指数位置则进行四舍五入处理。

    参数：
      expr: sympy 表达式
      in_exponent: 布尔型，指示当前 expr 是否处在幂（Pow）的指数位置
    返回：
      经过处理后的表达式
    """
    # 如果 expr 是原子对象（Atom），则直接返回，
    # 因为原子对象（包括 Symbol 或其他常量）不需要遍历其子节点
    if expr.is_Atom:
        if expr.is_number and expr.free_symbols == set() and not in_exponent:
            try:
                return round_if_needed(expr)
            except Exception:
                return expr
        return expr

    # 专门处理 Pow：保持指数原样，处理底数
    if expr.func == sp.Pow:
        base = process_expr(expr.args[0])
        exponent = process_expr(expr.args[1])  # 这里将指数也进行处理
        return sp.Pow(base, exponent)

    # 对于其它表达式，递归处理其所有子节点
    new_args = tuple(process_expr(arg, in_exponent=False) for arg in expr.args)
    return expr.func(*new_args)

def expr_to_func(sympy_expr, variables: List[str]):
    def cot(x):
        return 1 / np.tan(x)

    def acot(x):
        return 1 / np.arctan(x)

    def coth(x):
        return 1 / np.tanh(x)

    return lambdify(
        variables,
        sympy_expr,
        modules=["numpy", {"cot": cot, "acot": acot, "coth": coth}],
    )

def compute_r2(y_gt, y_pred):

    return r2_score(y_gt, y_pred)

def sample_points(func, num_vars, range_, target_noise):
    x = np.random.uniform(range_[0], range_[1], (200 , num_vars))
    # print(x.shape)
    y = evaluate_points(func, x[:,0:num_vars])

    y = np.squeeze(y)  # 保证 y 是一维
    is_valid = np.isfinite(y)
    x = x[is_valid]
    y = y[is_valid]

    if len(y) < 10:
        raise ValueError("Too few valid samples after removing NaN/Inf.")

    # 添加噪声
    scale = target_noise * np.sqrt(np.mean(np.square(y)))
    rng = np.random.RandomState()
    noise = rng.normal(loc=0.0, scale=scale, size=y.shape)
    y_noisy = y + noise

    res = np.concatenate((x, y_noisy.reshape(-1, 1)), axis=1)
    return res,y
def evaluate_points(func, points):
    y = func(*[points[:, i] for i in range(points.shape[1])])
    # print(y)
    y = np.reshape(y, (-1, 1))
    if y.shape[0] != points.shape[0]:
        y = np.broadcast_to(y, (points.shape[0], 1))
    if np.any(np.iscomplex(y)):
        return np.broadcast_to(np.inf, (points.shape[0], 1))
    return y.astype(np.float64)


import re


def get_variable_names(expr: str):
    # 找出所有变量 x_数字
    variables = re.findall(r'x_\d+', expr)

    # 去重并按编号排序
    unique_vars = sorted(set(variables), key=lambda x: int(x.split('_')[1]))

    return unique_vars
def pad_to_10_columns(tensor):
    # 检查形状是否为 [200, n]
    n = tensor.size(1)

    # 计算需要补多少列
    pad_cols = 10 - n

    # 用0补在右侧（dim=1）
    padded_tensor = F.pad(tensor, (0, pad_cols), mode='constant', value=0)

    return padded_tensor


@hydra.main(config_name="config",version_base = '1.1')
def main(cfg):
    # print(hydra.utils.to_absolute_path(cfg.val_path))
    test_data = load_metadata_hdf5("/home/lida/Desktop/visymre_no10v/scripts/data/val_data/10vars/500")
    # print(test_data.id2word)
    bfgs = BFGSParams(
        activated=cfg.inference.bfgs.activated,
        n_restarts=cfg.inference.bfgs.n_restarts,
        add_coefficients_if_not_existing=cfg.inference.bfgs.add_coefficients_if_not_existing,
        normalization_o=cfg.inference.bfgs.normalization_o,
        idx_remove=cfg.inference.bfgs.idx_remove,
        normalization_type=cfg.inference.bfgs.normalization_type,
        stop_time=cfg.inference.bfgs.stop_time,
    )
    params_fit = FitParams(word2id=test_data.word2id,
                           id2word=test_data.id2word,
                           una_ops=test_data.una_ops,
                           bin_ops=test_data.bin_ops,
                           total_variables=list(test_data.total_variables),
                           total_coefficients=list(test_data.total_coefficients),
                           rewrite_functions=list(test_data.rewrite_functions),
                           bfgs=bfgs,
                           beam_size=cfg.inference.beam_size,
                           device=cfg.inference.device

                           )
    print("test_data.word2id",test_data.word2id)
    model_path = hydra.utils.to_absolute_path(cfg.model_path)
    model = Model.load_from_checkpoint(model_path, cfg=cfg.architecture)
    model.eval()
    model.to(cfg.inference.device)
    fitfunc = partial(model.fitfunc2, cfg_params=params_fit)
    # 确保batch有效

    # expr = "2-2.1*cos(9.8*x_1)*sin(1.3*x_2)"
    expr = "1/(1+1/(x_2**4))+1/(1+1/(x_1**4))"
    range_ = [-1,1]
    sym_eq = sympify(expr)
    variables = get_variable_names(str(expr))

    lam = expr_to_func(sym_eq, variables)
    points, y_no_noise = sample_points(lam, len(variables), range_=range_, target_noise=0)
    points = torch.tensor(points)
    y_no_noise = torch.tensor(y_no_noise)

    # 注意：假设 pad_to_10_columns 可处理 points 的维度
    X = pad_to_10_columns(points[:, :-1]).to(cfg.inference.device)
    y = points[:, -1].to(cfg.inference.device)

    X_dict = {var: X[:, idx].cpu() for idx, var in enumerate(variables)}

    # 尝试不同的 beam_size 值，依次为 3, 10, 20
    # # print("X",X.shape)
    output = fitfunc(X, y, cfg_params=cfg.inference, test_data=test_data)

    pre_expr = (sp.sympify(output['best_bfgs_preds'][0]))
    print("pre_expr",pre_expr)
    y_pre = lambdify(",".join(variables), pre_expr)(**X_dict)
    r2_visymre = compute_r2(y_no_noise, y_pre)
    print("r2_visymre",r2_visymre)


if __name__ == "__main__":

    main()




