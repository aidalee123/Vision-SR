import os
import re
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sympy as sp
from sympy import preorder_traversal, sympify, lambdify
from functools import partial
import hydra
import warnings

try:
    from src.visymre.architectures.model import Model
    from src.visymre.dclasses import BFGSParams, FitParams
    from src.visymre.utils import load_metadata_hdf5, symbol_equivalence_single, coefficient_regularization
except ImportError:
    print("[Warning] visymre src modules not found. Ensure PYTHONPATH is set.")

warnings.filterwarnings("ignore")

def calculate_tree_size(expression_str):
    try:
        expr = sp.sympify(expression_str)
        nodes = list(preorder_traversal(expr))
        return len(nodes)
    except:
        return 0


def pad_to_10_columns(tensor):
    n = tensor.size(1)
    if n >= 10: return tensor[:, :10]
    return F.pad(tensor, (0, 10 - n), mode='constant', value=0)


def get_variable_names(expr_str):
    variables = re.findall(r'x_\d+', str(expr_str))
    return sorted(set(variables), key=lambda x: int(x.split('_')[1]))


def replace_variables(expression):
    expression = re.sub(r'\bx\b', 'x_1', str(expression))
    expression = re.sub(r'\by\b', 'x_2', expression)
    return expression


def round_if_needed(val):
    try:
        num = float(val)
        rounded = round(num, 1)
        if abs(rounded - int(rounded)) < 1e-10:
            return sp.Integer(int(rounded))
        else:
            return sp.Float(rounded)
    except:
        return val

def expr_to_func(sympy_expr, variables):

    def cot(x): return 1 / np.tan(x)

    def acot(x): return 1 / np.arctan(x)

    def coth(x): return 1 / np.tanh(x)

    return lambdify(variables, sympy_expr, modules=["numpy", {"cot": cot, "acot": acot, "coth": coth}])


def setup_visymre_model(cfg, metadata_path):
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")

    test_data = load_metadata_hdf5(metadata_path)

    bfgs = BFGSParams(
        activated=cfg.inference.bfgs.activated,
        n_restarts=cfg.inference.bfgs.n_restarts,
        add_coefficients_if_not_existing=cfg.inference.bfgs.add_coefficients_if_not_existing,
        normalization_o=cfg.inference.bfgs.normalization_o,
        idx_remove=cfg.inference.bfgs.idx_remove,
        normalization_type=cfg.inference.bfgs.normalization_type,
        stop_time=cfg.inference.bfgs.stop_time,
    )

    params_fit = FitParams(
        word2id=test_data.word2id, id2word=test_data.id2word,
        una_ops=test_data.una_ops, bin_ops=test_data.bin_ops,
        total_variables=list(test_data.total_variables),
        total_coefficients=list(test_data.total_coefficients),
        rewrite_functions=list(test_data.rewrite_functions),
        bfgs=bfgs, beam_size=cfg.inference.beam_size
    )

    model_path = hydra.utils.to_absolute_path(cfg.model_path)
    model = Model.load_from_checkpoint(model_path, cfg=cfg.architecture,map_location='cpu')
    model.eval()
    model.to(cfg.inference.device)
    fitfunc = partial(model.fitfunc2, cfg_params=params_fit)

    return fitfunc, test_data, params_fit

class IdentityScaler:

    def fit(self, X, y=None): pass

    def transform(self, X): return np.array(X)

    def inverse_transform(self, X): return np.array(X)

    def restore_x_expression(self, expr): return expr

    def restore_y_expression(self, expr): return expr


class AutoMagnitudeScaler:

    def __init__(self, centering=False):
        self.scales = None
        self.centering = centering

    def _round_scale_log_median(self, arr):
        arr = np.abs(arr)
        arr = arr[arr > 0]
        if len(arr) == 0: return 1.0
        log_median = np.median(np.log10(arr))
        return 10 ** int(np.floor(log_median))

    def fit(self, X, y=None):
        X = np.asarray(X)
        if X.ndim == 1:
            self.scales = self._round_scale_log_median(X)
        else:
            self.scales = np.array([self._round_scale_log_median(X[:, i]) for i in range(X.shape[1])])
        return self

    def transform(self, X):
        return np.array(X / self.scales, dtype=np.float32)

    def restore_x_expression(self, expr):
        if self.scales is None: return expr
        if isinstance(self.scales, (int, float)):
            return expr.subs({sp.Symbol("x_1"): sp.Symbol("x_1") / self.scales})
        else:
            subs_dict = {
                sp.Symbol(f"x_{i + 1}"): sp.Symbol(f"x_{i + 1}") / self.scales[i]
                for i in range(len(self.scales)) if self.scales[i] != 1.0
            }
            return expr.subs(subs_dict).simplify()

    def restore_y_expression(self, expr):
        if self.scales is None: return expr
        return expr * self.scales


class ZScoreScaler:
    def __init__(self):
        self.mean = 0.0;
        self.std = 1.0

    def fit(self, X, y=None):
        X = np.asarray(X)
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        if np.ndim(self.std) == 0:
            self.std = 1.0 if self.std == 0 else self.std
        else:
            self.std[self.std == 0] = 1.0
        return self

    def transform(self, X):
        return (X - self.mean) / self.std

    def restore_x_expression(self, expr):
        subs_dict = {}
        if np.ndim(self.mean) == 0:
            return expr.subs({sp.Symbol("x_1"): (sp.Symbol("x_1") - self.mean) / self.std})
        for i in range(len(self.mean)):
            sym = sp.Symbol(f"x_{i + 1}")
            subs_dict[sym] = (sym - self.mean[i]) / self.std[i]
        return expr.subs(subs_dict)

    def restore_y_expression(self, expr):
        return expr * self.std + self.mean


class MinMaxScaler:
    def __init__(self):
        self.min = 0.0;
        self.scale = 1.0

    def fit(self, X, y=None):
        X = np.asarray(X)
        self.min = np.min(X, axis=0)
        diff = np.max(X, axis=0) - self.min
        if np.ndim(diff) == 0:
            self.scale = 1.0 if diff == 0 else diff
        else:
            diff[diff == 0] = 1.0;
            self.scale = diff
        return self

    def transform(self, X):
        return (X - self.min) / self.scale

    def restore_x_expression(self, expr):
        subs_dict = {}
        if np.ndim(self.min) == 0:
            return expr.subs({sp.Symbol("x_1"): (sp.Symbol("x_1") - self.min) / self.scale})
        for i in range(len(self.min)):
            sym = sp.Symbol(f"x_{i + 1}")
            subs_dict[sym] = (sym - self.min[i]) / self.scale[i]
        return expr.subs(subs_dict)

    def restore_y_expression(self, expr):
        return expr * self.scale + self.min


def sample_points(func, num_vars, range_, target_noise):
    x = np.random.uniform(range_[0], range_[1], (200, num_vars))
    try:
        y = func(*[x[:, i] for i in range(x.shape[1])])
        y = np.reshape(y, (-1, 1))
        if y.shape[0] != x.shape[0]: y = np.broadcast_to(y, (x.shape[0], 1))
        if np.any(np.iscomplex(y)): y[:] = np.inf
    except:
        y = np.full((x.shape[0], 1), np.inf)

    y = np.squeeze(y).astype(np.float64)

    # Clean
    is_valid = np.isfinite(y)
    x = x[is_valid]
    y = y[is_valid]
    if len(y) < 10: raise ValueError(f"Too few valid samples: {len(y)}")

    # Noise
    scale = target_noise * np.sqrt(np.mean(np.square(y)))
    noise_val = np.random.normal(loc=0.0, scale=scale if scale > 0 else 0, size=y.shape)
    y_noisy = y + noise_val

    return np.concatenate((x, y_noisy.reshape(-1, 1)), axis=1), y

def optimize_expression_constants(expr, X_test_dict, y_test, max_iter=1000, lr=1e-2, device="cpu"):
    from sympy import Symbol

    class SymPyExpressionModule(nn.Module):
        def __init__(self, expr):
            super().__init__()
            self.original_expr = expr
            self.device = device
            self.variables = sorted([str(s) for s in expr.free_symbols])
            constants, symbol_map = [], {}
            expr_subs = expr

            def is_constant(node):
                return (node.is_Float or node.is_Number) and not node.is_Integer

            for node in preorder_traversal(expr):
                if is_constant(node) and node not in symbol_map:
                    sym = Symbol(f"param_{len(constants)}")
                    symbol_map[node] = sym
                    constants.append(float(node))
                    expr_subs = expr_subs.subs(node, sym)
            c1, c2 = Symbol(f"p_scale"), Symbol(f"p_bias")
            symbol_map["scale"], symbol_map["bias"] = c1, c2
            constants += [1.0, 0.0]

            self.symbol_map = list(symbol_map.values())
            self.expr = c1 * expr_subs + c2

            self.params = nn.ParameterList([
                nn.Parameter(torch.tensor(v, dtype=torch.float32, device=self.device))
                for v in constants
            ])

            expr_code = str(self.expr)
            for i, sym in enumerate(self.symbol_map):
                expr_code = expr_code.replace(str(sym), f"params[{i}]")
            lambda_code = f"lambda {', '.join(self.variables)}, params: {expr_code}"

            context = {
                "torch": torch,
                "sin": torch.sin, "cos": torch.cos, "tan": torch.tan,
                "asin": torch.arcsin, "acos": torch.arccos,
                "exp": torch.exp, "log": torch.log, "sqrt": torch.sqrt, "abs": torch.abs,
                "pi": torch.pi, "e": torch.exp(torch.tensor(1.0))
            }
            self._compiled = eval(lambda_code, context)

        def forward(self, input_dict):
            args = [input_dict[v].to(self.device) for v in self.variables]
            return self._compiled(*args, self.params)

        def to_sympy_expr(self):
            values = [float(p.detach().cpu()) for p in self.params]
            return self.expr.subs({sym: val for sym, val in zip(self.symbol_map, values)})

    model = SymPyExpressionModule(expr).to(device)
    input_dict = {k: v.float().to(device) for k, v in X_test_dict.items()}
    y_true = torch.tensor(y_test, dtype=torch.float32, device=device).view(-1)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    for i in range(max_iter):
        optimizer.zero_grad()
        loss = loss_fn(model(input_dict).view(-1), y_true)
        if not torch.isfinite(loss): break
        loss.backward()
        optimizer.step()
        if loss.item() < 1e-6: break


    return model.to_sympy_expr()
