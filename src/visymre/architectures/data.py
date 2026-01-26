from torch.utils import data
from src.visymre.utils import load_metadata_hdf5, load_eq
from sympy.core.rules import Transform
from sympy import sympify, Float, Symbol, Number, S, Integer, Float, symbols, sympify, solve, Eq, Symbol
from typing import List
from torch.distributions.uniform import Uniform
from ..dataset.data_utils import sample_symbolic_constants
from ..dataset.generator import Generator
import pytorch_lightning as pl
from src.visymre.dclasses import Equation
from functools import partial
from pathlib import Path
import hydra
import torch, random, math, warnings, re
from func_timeout import func_set_timeout, FunctionTimedOut, func_timeout
import logging
import numpy as np
import cv2
from sympy import lambdify
# 获取 logger 实例
logger = logging.getLogger(__name__)

SINGLE_EQ_TIMEOUT = 50  # [NEW] 单个方程计算超时时间(秒)

ALLOWED_INTS = {str(i) for i in range(-9, 10) if i != 0}

def get_random_orthogonal_basis(dim, rng=None):
    """
    Generate two random orthogonal unit vectors u, v in n-dim space.
    Samples uniformly from the Grassmannian manifold Gr(2, n).
    """
    if rng is None:
        rng = np.random

    # 1D Case
    if dim == 1:
        return np.array([1.0]), np.array([0.0])

    # Sample from standard normal distribution
    v1 = rng.randn(dim)
    v2 = rng.randn(dim)

    # Gram-Schmidt Orthogonalization
    norm_v1 = np.linalg.norm(v1) + 1e-8
    u = v1 / norm_v1

    v2_proj = v2 - np.dot(v2, u) * u
    norm_v2 = np.linalg.norm(v2_proj)

    # Handle edge case where v1 and v2 are parallel
    if norm_v2 < 1e-6:
        v2 = rng.randn(dim)
        v2_proj = v2 - np.dot(v2, u) * u
        norm_v2 = np.linalg.norm(v2_proj) + 1e-8

    v = v2_proj / norm_v2
    return u, v
# -----------------------------------------------------------------------------

# [NEW] Helper from data2.py
def contains_exp(expr_str):
    """
    判断字符串中是否包含 exp 或 log 函数调用
    例如 "exp(...)"、"((log(x_1)))" 等
    """
    pattern = r'\b(exp|log)\s*\('
    return bool(re.search(pattern, expr_str))


modules = {
    "numpy": np,
    "ln": np.log,
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "exp": np.exp,
    "log": np.log,
    "sqrt": np.sqrt,
    "Abs": np.abs,
    "pi": np.pi,
    "E": np.e,
    "asin": np.arcsin,
    "re": np.real  # 添加 're' 映射
}


# -----------------------------------------------------------------------------
# [NEW] Timeout Helper Functions
# -----------------------------------------------------------------------------
def run_with_timeout(func, args=(), kwargs=None, timeout=SINGLE_EQ_TIMEOUT):
    """
    通用超时包装器。如果在 timeout 秒内 func 未完成，抛出 FunctionTimedOut 异常。
    """
    if kwargs is None:
        kwargs = {}
    try:
        return func_timeout(timeout, func, args=args, kwargs=kwargs)
    except FunctionTimedOut:
        raise FunctionTimedOut(f"Execution timed out after {timeout}s")
    except Exception as e:
        raise e
class VisymresDataset(data.Dataset):
    def __init__(
            self,
            data_path: Path,
            cfg,
            mode: str
    ):
        # print(hydra.utils.to_absolute_path(data_path))
        metadata = load_metadata_hdf5(hydra.utils.to_absolute_path(data_path))
        cfg.total_variables = metadata.total_variables
        cfg.total_coefficients = metadata.total_coefficients
        self.len = metadata.total_number_of_eqs
        self.eqs_per_hdf = metadata.eqs_per_hdf
        self.word2id = metadata.word2id
        print(self.word2id)
        self.id2word = metadata.id2word
        # print(self.id2word)
        self.data_path = data_path
        self.mode = mode
        self.cfg = cfg

    def __getitem__(self, index):
        eq = load_eq(self.data_path, index, self.eqs_per_hdf)
        seed = index if self.mode != 'train' else None
        try:
            sympy_expr, t, _, eq_sympy_prefix = self.return_t_expr(eq)
            curr = Equation(expr=sympy_expr, coeff_dict={}, eq_sympy_prefix=eq_sympy_prefix,
                            variables=eq.variables,  # Uses variables from HDF5
                            support=eq.support, tokenized=t, valid=True)
        except FunctionTimedOut:
            curr = Equation(expr='x_1', coeff_dict={}, eq_sympy_prefix=[], variables=eq.variables,
                            support=eq.support, valid=False)
        except Exception as e:
            # logger.warning(f"Error in __getitem__: {e}")
            curr = Equation(expr='x_1', coeff_dict={}, eq_sympy_prefix=[], variables=eq.variables,
                            support=eq.support, valid=False)
        # [Modified] Attach seed to equation object to be used in plot_and_process
        curr.seed=seed

        return curr

    def __len__(self):
        return self.len
    @func_set_timeout(SINGLE_EQ_TIMEOUT * 2)  # Giving slightly more time for parsing
    def return_t_expr(self, eq):
        consts, initial_consts = sample_symbolic_constants(eq, self.cfg.constants)
        # print(consts)
        if self.cfg.predict_c:
            eq_string = eq.expr.format(**consts)
        else:
            eq_string = eq.expr.format(**initial_consts)
        eq_sympy_infix, sympy_expr = constants_to_placeholder(eq_string)

        eq_sympy_prefix = sanitize_prefix(Generator.sympy_to_prefix(eq_sympy_infix))
        # print("eq_pre",eq_sympy_prefix)
        t = tokenize(eq_sympy_prefix, self.word2id)
        return sympy_expr, t, consts, eq_sympy_prefix

def custom_collate_fn(eqs: List[Equation], cfg) -> List[torch.tensor]:
    filtered_eqs = [
        eq for eq in eqs
        if eq.valid
           and len(eq.tokenized) < 100
    ]

    res, tokens_eqs, fun_image, expr = evaluate_and_wrap(filtered_eqs, cfg)

    return [res, tokens_eqs, fun_image, expr]

def constants_to_placeholder(s, symbol="c"):

    sympy_expr = (sympify(s))
    # print("sympy_expr",sympy_expr)
    #sympy_expr = (sympify(sympy_expr))
    # print("sympy",sympy_expr)
    # save_to_csv(sympy_expr)
    eq_sympy_infix = sympy_expr.xreplace(
        Transform(
            lambda x: Symbol(symbol, real=True, nonzero=True),
            # 仅将浮点数或大整数替换为 c
            lambda x: isinstance(x, Float) or (isinstance(x, Integer) and abs(x) > 9)
        )
    )
    return eq_sympy_infix, sympy_expr

def extract_variables(equations):
    if not isinstance(equations, list):
        equations = [equations]
    variables = set()
    for eq in equations:
        symbols_in_eq = eq.free_symbols
        for s in symbols_in_eq:
            name = str(s)
            if re.fullmatch(r'x_\d+', name):
                variables.add(name)
    return sorted(variables, key=lambda x: int(x.split('_')[1]))

def sanitize_prefix(tokens):

    cleaned = []
    for t in tokens:
        # [新增] 处理虚数单位 I
        if t == 'I':
            cleaned.append('c')
            continue
        if t.lstrip("-").isdigit():
            cleaned.append(t if t in ALLOWED_INTS else "c")
            continue
        if re.fullmatch(r"[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?", t):
            cleaned.append("c")
            continue
        cleaned.append(t)
    return cleaned

def tokenize(prefix_expr: list, word2id: dict) -> list:
    tokenized_expr = []
    tokenized_expr.append(word2id["S"])
    for i in prefix_expr:
        tokenized_expr.append(word2id[i])
    tokenized_expr.append(word2id["F"])
    return tokenized_expr

def de_tokenize(tokenized_expr, id2word: dict):
    # print("token",tokenized_expr)
    prefix_expr = []
    for i in tokenized_expr:
        if isinstance(i, torch.Tensor):
            idx = i.item()
        else:
            idx = i

        if "F" == id2word[idx]:
            break
        else:
            prefix_expr.append(id2word[idx])
    # print("prefix_expr",prefix_expr)
    return prefix_expr
def tokens_padding(tokens):
    max_len = max([len(y) for y in tokens])
    p_tokens = torch.zeros(len(tokens), max_len)
    for i, y in enumerate(tokens):
        y = torch.tensor(y).long()
        p_tokens[i, :] = torch.cat([y, torch.zeros(max_len - y.shape[0]).long()])
    return p_tokens


def number_of_support_points(p, type_of_sampling_points):
    if type_of_sampling_points == "constant":
        curr_p = p
    elif type_of_sampling_points == "logarithm":
        curr_p = int(10 ** Uniform(1, math.log10(p)).sample())
    else:
        raise NameError
    return curr_p

def sample_support(curr_p, cfg, n_clusters):
    points_per_cluster = curr_p // n_clusters  # 每个簇中点的数量
    remainder = curr_p % n_clusters  # 处理余数
    cluster_samples = []
    for i in range(n_clusters):
        # 计算当前簇需要采样的点数
        curr_points_count = points_per_cluster + (1 if i < remainder else 0)
        while True:
            base_nums = random.choices(range(1, 11), k=2)
            bound1 = base_nums[0] * random.choice([-1, 1])
            bound2 = base_nums[1] * random.choice([-1, 1])
            if bound1 != bound2:
                break
        selected_min, selected_max = sorted((bound1, bound2))
        rand_mode = random.random()
        if rand_mode < 0.5:
            exponent_dist = torch.distributions.Uniform(float(selected_min), float(selected_max))
            exponents = exponent_dist.sample([int(curr_points_count)])
            points_curr = torch.exp(exponents)
        else:
            distribution = torch.distributions.Uniform(float(selected_min), float(selected_max))
            points_curr = distribution.sample([int(curr_points_count)])
        cluster_samples.append(points_curr)
    curr = torch.cat(cluster_samples)
    return curr


def var_sort_key(item):
    return int(item.split('_')[1])

def plot_and_process(eq, curr_p, n_clusters, cfg, plt_=None, seed=None):
    sorted_vars = tuple(sorted(eq.variables, key=str)) if len(eq.variables) > 1 else tuple(eq.variables)
    dim = len(sorted_vars)
    img_size = cfg.funimage_size
    N_CHANNELS = getattr(cfg, 'input_channels', 10)
    funimage = torch.zeros((img_size, img_size, N_CHANNELS), dtype=torch.float32)

    local_rng = np.random.RandomState(seed) if seed is not None else np.random
    scale_rng = np.random.RandomState(seed + 1000) if seed is not None else np.random

    num_vars = len(cfg.total_variables)
    support = torch.zeros((num_vars, curr_p), dtype=torch.float32)

    for i in range(dim):
        data = sample_support(curr_p, cfg, n_clusters)
        if isinstance(data, torch.Tensor):
            support[i, :] = data.float()
        else:
            support[i, :] = torch.from_numpy(data.astype(np.float32))

    if not plt_:
        return funimage, support
    try:
        if curr_p > 5:
            support_view = support[:dim, :].numpy()  # shared memory view
            # 指定 float32 计算
            center_mean = np.mean(support_view, axis=1, dtype=np.float32)
            max_std = np.max(np.std(support_view, axis=1, dtype=np.float32))
            base_sigma = max_std if max_std > 1e-4 else 1.0
        else:
            center_mean = np.zeros(dim, dtype=np.float32)
            base_sigma = 1.0
    except:
        center_mean = np.zeros(dim, dtype=np.float32)
        base_sigma = 1.0

    if N_CHANNELS == 1:
        random_power = scale_rng.uniform(np.log(0.2), np.log(5.0))
        scale_factors = np.array([np.exp(random_power)], dtype=np.float32)
    else:
        scale_factors = np.geomspace(0.2, 5.0, num=N_CHANNELS).astype(np.float32)

    try:
        f_numpy = lambdify(sorted_vars, eq.expr, modules='numpy')

        if dim == 1:
            num_points = 300
            canvas = np.zeros((img_size, img_size), dtype=np.float32)
            col_indices = np.linspace(0, img_size - 1, num_points).astype(np.int32)

            for ch in range(N_CHANNELS):
                r = (3.0 * base_sigma * scale_factors[ch]).astype(np.float32)

                if N_CHANNELS == 1:
                    c = 0.0 if scale_rng.rand() > 0.5 else center_mean[0]
                else:
                    c = 0.0 if ch < (N_CHANNELS // 2) else center_mean[0]

                x_vals = np.linspace(c - r, c + r, num_points, dtype=np.float32)

                try:
                    canvas.fill(0)

                    y_vals = f_numpy(x_vals)
                    if np.ndim(y_vals) == 0:
                        y_vals = np.full_like(x_vals, float(y_vals))
                    else:
                        y_vals = y_vals.astype(np.float32, copy=False)
                    np.nan_to_num(y_vals, copy=False, nan=0.0, posinf=1e30, neginf=-1e5)

                    y_min, y_max = np.min(y_vals), np.max(y_vals)
                    y_range = y_max - y_min

                    if y_range > 1e-6:
                        # 向量化计算坐标
                        norm_y = (y_vals - y_min) / y_range
                        y_indices = ((1.0 - norm_y) * (img_size - 1)).astype(np.int32)

                        pts = np.column_stack((col_indices, y_indices)).reshape((-1, 1, 2))
                        cv2.polylines(canvas, [pts], isClosed=False, color=1.0, thickness=2, lineType=cv2.LINE_AA)
                    else:
                        cv2.line(canvas, (0, img_size // 2), (img_size, img_size // 2), color=1.0, thickness=1)

                    funimage[:, :, ch] = torch.from_numpy(canvas)
                except:
                    pass
        else:
            s_norm = np.linspace(-1.0, 1.0, img_size, dtype=np.float32)
            S_flat_norm = np.tile(s_norm, img_size)
            T_flat_norm = np.repeat(s_norm, img_size)
            z_norm_buffer = np.zeros((img_size, img_size), dtype=np.float32)

            for ch in range(N_CHANNELS):
                current_radius = (base_sigma * 3.0 * scale_factors[ch]).astype(np.float32)
                # 利用广播乘法
                S_flat = S_flat_norm * current_radius
                T_flat = T_flat_norm * current_radius

                if seed is not None:
                    ch_rng = np.random.RandomState(seed * 100 + ch)
                else:
                    ch_rng = local_rng

                u, v = get_random_orthogonal_basis(dim, rng=ch_rng)
                u = u.astype(np.float32, copy=False)
                v = v.astype(np.float32, copy=False)

                # 中心策略
                if N_CHANNELS == 1:
                    use_center = np.zeros((dim, 1), dtype=np.float32) if scale_rng.rand() > 0.5 else center_mean[:,
                                                                                                     None]
                else:
                    use_center = np.zeros((dim, 1), dtype=np.float32) if ch < (N_CHANNELS // 2) else center_mean[:,
                                                                                                     None]

                X = use_center + np.outer(u, S_flat) + np.outer(v, T_flat)

                args = (X[i] for i in range(dim))

                try:
                    z_vals = f_numpy(*args)

                    if np.ndim(z_vals) == 0:
                        funimage[:, :, ch] = 0
                    else:
                        if np.iscomplexobj(z_vals): z_vals = z_vals.real
                        z_vals = z_vals.astype(np.float32, copy=False)

                        np.nan_to_num(z_vals, copy=False, nan=0.0, posinf=1e5, neginf=-1e5)

                        std_val = np.std(z_vals, dtype=np.float32)
                        scale_factor = std_val if std_val > 1e-6 else 1.0

                        np.arctan(z_vals / scale_factor, out=z_vals)

                        z_vals += 1.5707964
                        z_vals /= 3.1415927

                        np.clip(z_vals, 0.0, 1.0, out=z_vals)

                        funimage[:, :, ch] = torch.from_numpy(z_vals.reshape(img_size, img_size))

                except Exception:
                    pass

    except Exception:
        pass

    return funimage, support

def return_y(eq, support):
    eq_numpy = lambdify(sorted(list(eq.variables), key=var_sort_key), eq.expr, modules=modules)
    if len(eq.variables) == 1:
        y = eq_numpy(support[0, :])
    else:
        y = eq_numpy(*support[0:len(eq.variables), :])
    if np.random.rand() < 1:  # 100%的几率添加噪声
        target_noise = random.uniform(0, 0.1)
        valid_y = y[~torch.isnan(y)]
        if valid_y.numel() > 0:  # 确保有有效数据
            scale = target_noise * torch.sqrt(torch.mean(torch.square(valid_y))).item()
            if np.iscomplex(scale):
                scale = scale.real
            noise = np.random.normal(loc=0.0, scale=scale, size=y.shape)
            # 对于有 nan 的位置，噪声也设为 nan
            noise[torch.isnan(y)] = np.nan
            y = torch.tensor(y + noise, dtype=torch.float32)
        else:
            y = torch.tensor(y, dtype=torch.float32)
    return y, eq_numpy

def _safe_processing_logic(eq, curr_p, cfg, plt_, seed):
    """
    纯函数逻辑：生成 Support, Image 和 Y。
    用于被 func_timeout 包装。
    """
    image, support = plot_and_process(eq, curr_p, n_clusters=cfg.n_clusters, cfg=cfg, plt_=plt_, seed=seed)
    y, _ = return_y(eq, support)
    return image, support, y

def _sample_once(eq, curr_p, cfg, plt_=None):
    """
    单次采样 → (image, support, y, invalid_indices, success_flag)
    增加了 run_with_timeout 保护
    """
    try:
        seed = getattr(eq, 'seed', None)

        # [TIMEOUT CHECK] 限制单个方程的计算时间
        image, support, y = run_with_timeout(
            _safe_processing_logic,
            args=(eq, curr_p, cfg, plt_, seed),
            timeout=SINGLE_EQ_TIMEOUT
        )

        success = False
        invalid_indices = torch.tensor([], dtype=torch.long)

        if isinstance(y, torch.Tensor) and y.dtype == torch.float32:
            y = y.squeeze(0)
            invalid_indices = torch.where(
                torch.isnan(y) | torch.isinf(y) | (abs(y) > cfg.eps_limit)
            )[0]
            success = (len(invalid_indices) <= curr_p * 0.5)

        return image, support, y, invalid_indices, success

    except FunctionTimedOut:
        logger.warning(f"Timeout skipping eq: {str(eq.expr)[:30]}...")
        return [], [], [], [], False
    except Exception as e:
        return [], [], [], [], False

def evaluate_and_wrap(eqs: List[Equation], cfg):
    vals = []
    cond0 = []
    fun_image = []

    expr = [eq.expr for eq in eqs]
    tokens_eqs = [eq.tokenized for eq in eqs]
    tokens_eqs = tokens_padding(tokens_eqs)

    curr_p = number_of_support_points(cfg.max_number_of_points, cfg.type_of_sampling_points)

    for eq in eqs:
        # print("ewq",eq.variables)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for retry in range(cfg.max_retry):
                _, support, y, invalid_idx, ok = _sample_once(eq, curr_p, cfg, plt_=False)

                if ok:

                    try:
                        seed = getattr(eq, 'seed', None)
                        image, _ = run_with_timeout(
                            plot_and_process,
                            args=(eq, curr_p, cfg.n_clusters, cfg, True, seed),
                            timeout=SINGLE_EQ_TIMEOUT
                        )
                        break  # 成功，跳出循环
                    except (FunctionTimedOut, Exception) as e:
                        # 图像生成失败，继续重试
                        continue
            else:
                cond0.append(False)
                continue
            # print(cond0)
            fun_image.append(image)

            y_fixed = y.clone()
            if invalid_idx.numel() > 0:
                y_fixed[invalid_idx] = 0
                support[:, invalid_idx] = 0

            concatenated = torch.cat([support, y_fixed.unsqueeze(0)], dim=0)
            vals.append(concatenated.unsqueeze(0))
            cond0.append(True)
    if not cond0:
        raise RuntimeError("All equations in batch failed or timed out.")
    cond0_tensor = torch.tensor(cond0, dtype=torch.bool)

    fun_image = torch.stack(fun_image, dim=0)
    res = torch.cat(vals, dim=0)

    expr = [e for i, e in enumerate(expr) if cond0[i]]
    tokens_eqs = tokens_eqs[cond0_tensor]
    return res, tokens_eqs, fun_image, expr


class DataModule(pl.LightningDataModule):
    def __init__(
            self,
            data_train_path,
            data_val_path,
            data_test_path,
            cfg,
    ):
        super().__init__()
        self.cfg = cfg
        self.data_train_path = data_train_path
        self.data_val_path = data_val_path
        self.data_test_path = data_test_path

    def setup(self, stage=None):
        """called one ecah GPU separately - stage defines if we are at fit or test step"""
        if stage == "fit" or stage is None:
            if self.data_train_path:
                self.training_dataset = VisymresDataset(
                    self.data_train_path,
                    self.cfg.dataset_train,
                    mode="train"
                )
            if self.data_val_path:
                self.validation_dataset = VisymresDataset(
                    self.data_val_path,
                    self.cfg.dataset_val,
                    mode="val"
                )
            # if stage == 'test' or stage is None:
            if self.data_test_path:
                self.test_dataset = VisymresDataset(
                    self.data_test_path, self.cfg.dataset_test,
                    mode="test"
                )

    def train_dataloader(self):
        """returns training dataloader"""
        trainloader = torch.utils.data.DataLoader(
            self.training_dataset,
            batch_size=self.cfg.architecture.batch_size,
            shuffle=True,
            # drop_last=True,
            collate_fn=partial(custom_collate_fn, cfg=self.cfg.dataset_train),
            num_workers=self.cfg.num_of_workers,
            pin_memory=True,
            persistent_workers=True
        )
        return trainloader

    def val_dataloader(self):
        """returns validation dataloader"""
        validloader = torch.utils.data.DataLoader(
            self.validation_dataset,
            batch_size=self.cfg.architecture.batch_size,
            shuffle=False,
            collate_fn=partial(custom_collate_fn, cfg=self.cfg.dataset_val),
            num_workers=self.cfg.num_of_workers,
            pin_memory=True,
            drop_last=False
        )
        return validloader

    def test_dataloader(self):

        """returns validation dataloader"""
        testloader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=partial(custom_collate_fn, cfg=self.cfg.dataset_test),
            num_workers=0,
            pin_memory=True,
            drop_last=False
        )
        return testloader
