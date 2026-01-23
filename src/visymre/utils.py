import marshal
import random
import sympy as sp
import pickle
from .dclasses import DatasetDetails, Equation
from typing import List, Tuple
import h5py
import os
import numpy as np
from pathlib import Path
import re
MAXTIME = 60

import signal,sympy
class SimplifyTimeOutException(Exception):
    pass

def alarm_handler(signum, frame):
    print(f"raising SimplifyTimeOutException")
    raise SimplifyTimeOutException

def solve_and_swap_random(expr):

    free_syms = list(expr.free_symbols)
    if not free_syms:
        return expr

    target_var = random.choice(free_syms)

    try:
        shadow = sympy.Symbol('__SHADOW__')

        solutions = sympy.solve(
            sympy.Eq(shadow, expr),
            target_var,
            manual=True,
            check=False,
            simplify=False
        )


        if not solutions:
            return expr

        valid_sols = [s for s in solutions if 'I' not in str(s)]

        candidates = valid_sols if valid_sols else solutions

        chosen_sol = random.choice(candidates)

        final_expr = chosen_sol.subs(shadow, target_var)

        return final_expr

    except Exception:

        return expr

class AutoMagnitudeScaler:

    def __init__(self, verbose=False, centering=False):
        self.scales = None
        self.centers = None  # [NEW] 存储中位数
        self.centering = centering  # [NEW] 开关
        self.is_bounded_detected = False
        self.verbose = verbose
        self.diagnostics = {}
    @staticmethod
    def _calculate_robust_params(arr, centering=False):
        """同时计算 Center 和 Scale"""
        arr = np.asarray(arr)
        # 排除 NaN/Inf
        arr = arr[np.isfinite(arr)]

        if len(arr) == 0: return 0.0, 1.0

        med_val = np.median(arr)
        center = med_val if centering else 0.0

        arr_centered = arr - center
        arr_abs = np.abs(arr_centered)
        arr_nonzero = arr_abs[arr_abs > 0]

        q75, q25 = np.percentile(arr, [75, 25])
        iqr = q75 - q25

        abs_med = np.median(arr_nonzero) if len(arr_nonzero) > 0 else 1.0

        if iqr > 1e-12:
            target_metric = iqr
        else:
            target_metric = abs_med

        if target_metric < 1e-300: target_metric = 1.0

        log_val = np.log10(target_metric)
        exponent = int(np.floor(log_val))
        exponent = np.clip(exponent, -300, 300)

        if abs(exponent) >= 1:
            scale = 10.0 ** float(exponent)
        else:
            scale = 1.0

        return center, scale

    def fit(self, X, y=None):
        X = np.asarray(X)
        self.is_bounded_detected = False
        self.diagnostics = {}

        if X.ndim == 1:
            c, s = self._calculate_robust_params(X, self.centering)
            self.centers = c
            raw_scales = s
        else:
            params = [self._calculate_robust_params(X[:, i], self.centering) for i in range(X.shape[1])]
            self.centers = np.array([p[0] for p in params])
            raw_scales = np.array([p[1] for p in params])

        self.diagnostics["raw_scales"] = raw_scales
        self.diagnostics["centers"] = self.centers

        perform_stability_check = (y is not None) and (X.ndim > 1)

        if perform_stability_check:
            is_stable, details = self._check_is_stable_function(X, np.asarray(y))
            self.diagnostics.update(details)

            if is_stable:
                if X.ndim == 1:
                    self.scales = 1.0
                else:
                    self.scales = np.ones(X.shape[1])
                # 如果判定为稳定函数，也不去中心化，保持原样
                if np.ndim(self.centers) == 0:
                    self.centers = 0.0
                else:
                    self.centers = np.zeros(X.shape[1])

                self.is_bounded_detected = True
                self.diagnostics["final_decision"] = "Bounded -> Force Identity"
            else:
                self.scales = raw_scales
                self.diagnostics["final_decision"] = "Unbounded -> Robust Scale"
        else:
            self.scales = raw_scales
            self.diagnostics["final_decision"] = "Standard Robust Scale"

        self.diagnostics["final_scales"] = self.scales

        if self.verbose: self._print_diagnostics()
        return self

    def _check_is_stable_function(self, X, y):

        return False, {"score": 999}
    def _print_diagnostics(self):
        d = self.diagnostics
        print(f"\n[AutoMagnitudeScaler Diagnostics]")
        print(f"  > Center (Bias): {d.get('centers')}")  # [NEW]
        print(f"  > Scale (Spread): {d.get('raw_scales')}")
        print(f"  > Final Decision: {d['final_decision']}")

    def transform(self, X):
        if self.scales is None: raise ValueError("Scaler not fit.")
        X = np.asarray(X)
        # (X - Center) / Scale
        return ((X - self.centers) / self.scales).astype(np.float32)

    def inverse_transform(self, X_scaled):
        if self.scales is None: raise ValueError("Scaler not fit.")
        # X * Scale + Center
        return (np.asarray(X_scaled) * self.scales + self.centers).astype(np.float32)

    def restore_x_expression(self, expr):

        if self.scales is None: return expr

        # 1D Case
        if np.ndim(self.scales) == 0:
            s = self.scales
            c = self.centers
            if s == 1.0 and c == 0.0: return expr
            return expr.subs({sp.Symbol("x_1"): (sp.Symbol("x_1") - c) / s})

        # ND Case
        subs_dict = {}
        for i, (s, c) in enumerate(zip(self.scales, self.centers)):
            if s != 1.0 or c != 0.0:
                sym = sp.Symbol(f"x_{i + 1}")
                subs_dict[sym] = (sym - c) / s
        return expr.subs(subs_dict)

    def restore_y_expression(self, expr):

        if self.scales is None: return expr
        s = self.scales
        c = self.centers
        if isinstance(s, (np.ndarray, list)):
            s = s[0] if len(s) > 0 else 1.0
            c = c[0] if len(c) > 0 else 0.0

        if s == 1.0 and c == 0.0: return expr
        # y_raw = y_scaled * s + c
        return expr * s + c


class H5FilesCreator():
    def __init__(self,base_path: Path = None, target_path: Path = None, metadata=None):

        target_path.mkdir(mode=0o777, parents=True, exist_ok=True)
        self.target_path = target_path
        
        self.base_path = base_path
        self.metadata = metadata
        

    def create_single_hd5_from_eqs(self,block):
        name_file, eqs = block
        t_hf = h5py.File(os.path.join(self.target_path, str(name_file) + ".h5") , 'w')
        for i, eq in enumerate(eqs):            
            curr = np.void(pickle.dumps(eq))
            t_hf.create_dataset(str(i), data=curr)
        t_hf.close()
    
    def recreate_single_hd5_from_idx(self,block:Tuple):
        name_file, eq_idxs = block
        t_hf = h5py.File(os.path.join(self.target_path, str(name_file) + ".h5") , 'w')
        for i, eq_idx in enumerate(eq_idxs):            
            eq = load_eq_raw(self.base_path, eq_idx, self.metadata.eqs_per_hdf)
            #curr = np.void(pickle.dumps(eq))
            t_hf.create_dataset(str(i), data=eq)
        t_hf.close()


def code_unpickler(data):
    return marshal.loads(data)

def code_pickler(code):
    return code_unpickler, (marshal.dumps(code),)

def load_eq_raw(path_folder, idx, num_eqs_per_set) -> Equation:
    index_file = str(int(idx/num_eqs_per_set))
    f = h5py.File(os.path.join(path_folder,f"{index_file}.h5"), 'r')
    dataset_metadata = f[str(idx - int(index_file)*int(num_eqs_per_set))]
    raw_metadata = np.array(dataset_metadata)
    f.close()
    return raw_metadata

def load_eq(path_folder, idx, num_eqs_per_set) -> Equation:
    index_file = str(int(idx/num_eqs_per_set))
    f = h5py.File(os.path.join(path_folder,f"{index_file}.h5"), 'r')
    dataset_metadata = f[str(idx - int(index_file)*int(num_eqs_per_set))]
    raw_metadata = np.array(dataset_metadata)
    metadata = pickle.loads(raw_metadata.tobytes())
    f.close()
    return metadata

def load_metadata_hdf5(path_folder: Path) -> DatasetDetails:
    f = h5py.File(os.path.join(path_folder,"metadata.h5"), 'r')
    # print(f)
    dataset_metadata = f["other"]
    raw_metadata = np.array(dataset_metadata)

    metadata = pickle.loads(raw_metadata.tobytes())
    return metadata

def alarm_handler(signum, frame):
    print("raising SimplifyTimeOutException")
    raise SimplifyTimeOutException


def round_floats(expr):
    expr_mod = expr
    for a in sp.preorder_traversal(expr):
        if isinstance(a, sp.Float):
            if abs(a) < 0.0001:
                expr_mod = expr_mod.subs(a, sp.Integer(0))
            else:
                expr_mod = expr_mod.subs(a, round(a, 3))
    return expr_mod


def get_symbolic_model(expr_str, local_dict):
    sp_model = sp.parse_expr(expr_str, local_dict=local_dict)
    sp_model = round_floats(sp_model)
    signal.signal(signal.SIGALRM, alarm_handler)
    signal.alarm(MAXTIME)
    try:
        sp_model = sp.simplify(sp_model)
    except Exception as e:
        print('Warning: simplify failed. Msg:', e)
    finally:
        signal.alarm(0)
    return sp_model
def regularization(match):

    num_str = match.group()
    try:

        x = float(num_str)
    except ValueError:
        return num_str

    candidates = [
        (0, 0.1),
        (1, 0.01),
        (2, 0.001),
        (3, 0.0001)
    ]

    for digits, thresh in candidates:
        rounded = round(x, digits)
        if abs(x - rounded) <= thresh:

            if digits == 0:
                return str(int(rounded))
            else:

                return f"{rounded:.{digits}f}"

    return num_str

def coefficient_regularization(expression):
    """
    扫描表达式中的常数，按约定规则进行替换，并返回修改后的表达式。
    """

    pattern = r'(?<![A-Za-z_])[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?'

    new_expression = re.sub(pattern, regularization, expression)
    return new_expression

def symbolic_equivalence(true_model_expr, pred_model_str, local_dict):
    """
    判断预测的表达式（字符串形式）是否与给定的真值表达式（sympy 对象）等价。
    返回1表示等价，返回0表示不等价。
    """
    sp_model = get_symbolic_model(pred_model_str, local_dict)
    sym_diff = round_floats(true_model_expr - sp_model)
    sym_frac = round_floats(sp_model / true_model_expr)
    print('true_model:', true_model_expr, '; \npred_model:', sp_model)
    try:
        diff_const = sym_diff.is_constant(simplify=False)
        frac_const = sym_frac.is_constant(simplify=False)
        if not diff_const and not frac_const:
            signal.signal(signal.SIGALRM, alarm_handler)
            signal.alarm(MAXTIME)
            try:
                if not diff_const:
                    sym_diff = sp.simplify(sym_diff)
                    diff_const = sym_diff.is_constant()
                if not frac_const:
                    sym_frac = sp.simplify(sym_frac)
                    frac_const = sym_frac.is_constant()
            except Exception as e:
                print('Warning: simplify failed. Msg:', e)
            finally:
                signal.alarm(0)
    except Exception as e:
        print('Constant checking failed.', e)
        diff_const = False
        frac_const = False
    is_equivalent = (str(sym_diff) == '0' or diff_const or frac_const)
    return 1 if is_equivalent else 0


def symbol_equivalence_single(true_model_str, pred_model_str, feature_names):

    local_dict = {f: sp.Symbol(f) for f in feature_names}
    try:
        true_expr = get_symbolic_model(true_model_str, local_dict)
    except Exception as e:
        print(f"解析真值表达式失败: {true_model_str}，错误: {e}")
        return 0
    return symbolic_equivalence(true_expr, pred_model_str, local_dict)

