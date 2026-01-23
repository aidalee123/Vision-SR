import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import sympy as sp
import cv2
from sklearn.metrics import r2_score
import torch.nn.functional as F
import hashlib

# 引入项目原生模块
from src.visymre.architectures import bfgs


# --- 安全数学函数定义 ---
def safe_log(x):
    return np.log(np.abs(x) + 1e-6)


def safe_sqrt(x):
    return np.sqrt(np.abs(x) + 1e-6)


SAFE_MODULES = {
    "numpy": np,
    "ln": safe_log,
    "log": safe_log,
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "exp": np.exp,
    "sqrt": safe_sqrt,
    "Abs": np.abs,
    "abs": np.abs,
    "pi": np.pi,
    "E": np.e,
    "asin": np.arcsin,
    "re": np.real
}


# ==============================================================================
# 2. Holographic Renderer
# ==============================================================================
def get_random_orthogonal_basis(dim, rng=None):
    if rng is None: rng = np.random
    if dim == 1: return np.array([1.0]), np.array([0.0])
    v1 = rng.randn(dim);
    v2 = rng.randn(dim)
    norm_v1 = np.linalg.norm(v1) + 1e-8
    u = v1 / norm_v1
    v2_proj = v2 - np.dot(v2, u) * u
    norm_v2 = np.linalg.norm(v2_proj)
    if norm_v2 < 1e-6:
        v2 = rng.randn(dim)
        v2_proj = v2 - np.dot(v2, u) * u
        norm_v2 = np.linalg.norm(v2_proj) + 1e-8
    v = v2_proj / norm_v2
    return u, v


class HolographicRenderer:
    def __init__(self, img_size=112, input_channels=10):
        self.img_size = img_size
        self.channels = input_channels
        s = np.linspace(-1.0, 1.0, img_size, dtype=np.float32)
        self.S_flat_norm = np.tile(s, img_size)
        self.T_flat_norm = np.repeat(s, img_size)

    def render(self, sympy_expr, X_effective, active_indices, total_var_count=10):
        dim = X_effective.shape[1]
        all_vars = [sp.Symbol(f'x_{i + 1}') for i in range(total_var_count)]

        try:
            f_numpy = sp.lambdify(all_vars, sympy_expr, modules=SAFE_MODULES)
        except Exception:
            return torch.zeros((1, self.channels, self.img_size, self.img_size))

        if dim > 0:
            center_mean = np.mean(X_effective, axis=0, dtype=np.float32)
            max_std = np.max(np.std(X_effective, axis=0, dtype=np.float32))
            base_sigma = max_std if max_std > 1e-4 else 1.0
        else:
            center_mean = np.zeros(dim, dtype=np.float32)
            base_sigma = 1.0

        funimage = torch.zeros((self.img_size, self.img_size, self.channels), dtype=torch.float32)
        scale_factors = np.geomspace(0.2, 5.0, num=self.channels).astype(np.float32)

        if dim == 1:
            num_points = 150
            canvas = np.zeros((self.img_size, self.img_size), dtype=np.float32)
            col_indices = np.linspace(0, self.img_size - 1, num_points).astype(np.int32)
            for ch in range(self.channels):
                try:
                    r = (3.0 * base_sigma * scale_factors[ch]).astype(np.float32)
                    c = 0.0 if ch < (self.channels // 2) else center_mean[0]
                    x_slice_vals = np.linspace(c - r, c + r, num_points, dtype=np.float32)
                    full_args = [np.zeros_like(x_slice_vals) for _ in range(total_var_count)]
                    full_args[active_indices[0]] = x_slice_vals

                    y_vals = f_numpy(*full_args)
                    if np.ndim(y_vals) == 0: y_vals = np.full_like(x_slice_vals, float(y_vals))
                    if np.iscomplexobj(y_vals): y_vals = np.abs(y_vals)
                    y_vals = y_vals.astype(np.float32)

                    np.nan_to_num(y_vals, copy=False, nan=0.0, posinf=1e5, neginf=-1e5)
                    y_min, y_max = np.min(y_vals), np.max(y_vals)
                    y_range = y_max - y_min

                    canvas.fill(0)
                    if y_range > 1e-6:
                        norm_y = (y_vals - y_min) / y_range
                        y_indices = ((1.0 - norm_y) * (self.img_size - 1)).astype(np.int32)
                        pts = np.column_stack((col_indices, y_indices)).reshape((-1, 1, 2))
                        cv2.polylines(canvas, [pts], isClosed=False, color=1.0, thickness=1, lineType=cv2.LINE_AA)
                    else:
                        cv2.line(canvas, (0, self.img_size // 2), (self.img_size, self.img_size // 2), color=1.0,
                                 thickness=1)
                    funimage[:, :, ch] = torch.from_numpy(canvas)
                except:
                    pass
        else:
            for ch in range(self.channels):
                try:
                    r = (3.0 * base_sigma * scale_factors[ch]).astype(np.float32)
                    S_flat = self.S_flat_norm * r
                    T_flat = self.T_flat_norm * r
                    u, v = get_random_orthogonal_basis(dim)
                    u, v = u.astype(np.float32), v.astype(np.float32)
                    use_center = np.zeros(dim, dtype=np.float32) if ch < (self.channels // 2) else center_mean
                    X_slice_eff = use_center[None, :] + np.outer(S_flat, u) + np.outer(T_flat, v)

                    full_args = []
                    eff_col_idx = 0
                    for i in range(total_var_count):
                        if i in active_indices:
                            full_args.append(X_slice_eff[:, eff_col_idx])
                            eff_col_idx += 1
                        else:
                            full_args.append(np.zeros(X_slice_eff.shape[0], dtype=np.float32))

                    z_vals = f_numpy(*full_args)
                    if np.ndim(z_vals) == 0:
                        funimage[:, :, ch] = 0
                        continue
                    if np.iscomplexobj(z_vals): z_vals = np.abs(z_vals)
                    z_vals = z_vals.astype(np.float32)

                    np.nan_to_num(z_vals, copy=False, nan=0.0, posinf=1e5, neginf=-1e5)
                    std_val = np.std(z_vals)
                    scale = std_val if std_val > 1e-6 else 1.0
                    z_vals = np.arctan(z_vals / scale)
                    z_vals += 1.5707964
                    z_vals /= 3.1415927
                    np.clip(z_vals, 0.0, 1.0, out=z_vals)
                    funimage[:, :, ch] = torch.from_numpy(z_vals.reshape(self.img_size, self.img_size))
                except:
                    pass

        return funimage.permute(2, 0, 1).unsqueeze(0)


# ==============================================================================
# 3. LSTM Student
# ==============================================================================
class LSTMStudent(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_seq):
        emb = self.embedding(input_seq)
        output, (hn, cn) = self.lstm(emb)
        last_step_output = output[:, -1, :]
        logits = self.fc(last_step_output)
        return logits


# ==============================================================================
# 4. HLSC Class (Final DSR Optimized)
# ==============================================================================
class HolographicSelfCorrection:
    def __init__(self, model, test_data, cfg, device='cuda'):
        self.top_k = cfg.top_k
        self.model = model
        self.test_data = test_data
        self.cfg = cfg
        self.device = device
        self.renderer = HolographicRenderer(img_size=112, input_channels=model.cfg.input_channels)

        self.word2id = test_data.word2id
        self.id2word = test_data.id2word
        self.pad_idx = model.trg_pad_idx

        self.sos_idx = self.word2id['S']
        self.eos_idx = self.word2id['F']

        self.expression_cache = {}

        if 3 not in self.id2word:
            self.id2word[3] = "constant"
        elif self.id2word[3] != "constant":
            self.id2word[3] = "constant"

        # --- Prefix DSR: 构建 Token 元数表 ---
        self.arity_map = self._build_arity_map()

        # [优化] 预计算 Mask Tensors (移到这里，只计算一次)
        self._precompute_mask_tensors()

        # --- 构建结构约束表 (Nesting Constraints) ---
        self._build_structural_constraints()

    def _build_arity_map(self):
        arity_map = {
            'arity_2': [],
            'arity_1': [],
            'arity_0': [],  # Terminals
            'eos': [self.eos_idx],
            'ignore': [self.sos_idx, self.pad_idx]
        }

        bin_ops = ['add', 'sub', 'mul', 'div', 'pow']
        una_ops = ['abs', 'asin', 'cos', 'exp', 'ln', 'sin', 'sqrt', 'tan']

        for word, idx in self.word2id.items():
            if idx in [self.sos_idx, self.eos_idx, self.pad_idx]:
                continue

            if word in bin_ops:
                arity_map['arity_2'].append(idx)
            elif word in una_ops:
                arity_map['arity_1'].append(idx)
            else:
                arity_map['arity_0'].append(idx)

        return arity_map

    def _precompute_mask_tensors(self):
        """
        [优化] 预先在 GPU 上分配好固定的 Mask Index Tensor
        """
        arity_2 = set(self.arity_map['arity_2'])
        arity_1 = set(self.arity_map['arity_1'])
        arity_0 = set(self.arity_map['arity_0'])
        eos = set(self.arity_map['eos'])

        # 转换为 list 确保顺序，然后转 tensor
        self.mask_force_eos = torch.tensor(list(arity_2 | arity_1 | arity_0), device=self.device, dtype=torch.long)
        self.mask_ban_eos = torch.tensor(list(eos), device=self.device, dtype=torch.long)
        self.mask_force_terminal = torch.tensor(list(arity_2 | arity_1 | eos), device=self.device, dtype=torch.long)

        # 用于快速更新 stack
        self.arity_2_set = arity_2
        self.arity_1_set = arity_1
        self.arity_0_set = arity_0

    def _build_structural_constraints(self):
        """
        预计算需要禁止嵌套的 Operator IDs
        """
        nested_candidates = {'sin', 'cos', 'exp', 'ln', 'log'}
        self.nested_ban_ids = []
        for op in nested_candidates:
            if op in self.word2id:
                self.nested_ban_ids.append(self.word2id[op])

        self.pow_id = -1
        if 'pow' in self.word2id:
            self.pow_id = self.word2id['pow']

        self.nested_ban_tensor = torch.tensor(self.nested_ban_ids, device=self.device, dtype=torch.long)
        self.pow_ban_tensor = torch.tensor([self.pow_id] if self.pow_id != -1 else [], device=self.device,
                                           dtype=torch.long)
        self.nested_ban_set = set(self.nested_ban_ids)

    def _detect_unused_variables(self, X_tensor):
        X = X_tensor[0]
        unused_ids = []
        num_vars = X.shape[1]

        std_vals = torch.std(X, dim=0)
        mean_abs_vals = torch.mean(torch.abs(X), dim=0)
        threshold = 1e-6

        for i in range(num_vars):
            if std_vals[i] < threshold and mean_abs_vals[i] < threshold:
                var_name = f"x_{i + 1}"
                if var_name in self.word2id:
                    unused_ids.append(self.word2id[var_name])

        return unused_ids

    def _get_prefix_mask(self, curr_tokens, batch_size, vocab_size, max_len, ban_ids=None):
        """
        [优化] 直接使用预计算的 self.mask_xxx Tensors
        """
        mask = torch.zeros((batch_size, vocab_size), device=self.device)

        # 1. 屏蔽未使用的变量
        if ban_ids is not None and len(ban_ids) > 0:
            mask[:, ban_ids] = -float('inf')

        # 2. 语义栈模拟 (Deep Structural Constraints)
        curr_tokens_list = curr_tokens.cpu().tolist()

        row_nested_ban = []
        row_pow_ban = []

        for i in range(batch_size):
            stack = []
            seq = curr_tokens_list[i]
            for token in seq:
                if token == self.sos_idx or token == self.pad_idx: continue
                if token == self.eos_idx:
                    stack = []
                    break

                if stack:
                    stack[-1][1] -= 1

                if token in self.arity_2_set:
                    stack.append([token, 2])
                elif token in self.arity_1_set:
                    stack.append([token, 1])

                while stack and stack[-1][1] == 0:
                    stack.pop()

            if not stack: continue

            ancestors = [item[0] for item in stack]
            if any(aid in self.nested_ban_set for aid in ancestors):
                row_nested_ban.append(i)

            parent_id, parent_rem = stack[-1]
            if self.pow_id != -1 and parent_id == self.pow_id and parent_rem == 2:
                row_pow_ban.append(i)

        if len(self.nested_ban_ids) > 0 and len(row_nested_ban) > 0:
            rows = torch.tensor(row_nested_ban, device=self.device, dtype=torch.long)
            mask[rows[:, None], self.nested_ban_tensor] = -float('inf')

        if self.pow_id != -1 and len(row_pow_ban) > 0:
            rows = torch.tensor(row_pow_ban, device=self.device, dtype=torch.long)
            mask[rows[:, None], self.pow_ban_tensor] = -float('inf')

        # 3. Stack Count Logic (并行计算)
        current_lengths = curr_tokens.shape[1]
        slots_list = []
        for i in range(batch_size):
            slots = 1
            seq = curr_tokens[i].cpu().tolist()
            for token in seq:
                if token == self.sos_idx or token == self.pad_idx: continue
                if token == self.eos_idx:
                    slots = 0
                    break
                if token in self.arity_2_set:
                    slots += 1
                elif token in self.arity_1_set:
                    slots += 0
                elif token in self.arity_0_set:
                    slots -= 1
            slots_list.append(slots)

        slots_tensor = torch.tensor(slots_list, device=self.device)

        # [优化] 使用预计算的 Tensors
        finished_mask = (slots_tensor == 0)
        if finished_mask.any():
            rows = torch.nonzero(finished_mask).squeeze(1)
            mask[rows[:, None], self.mask_force_eos] = -float('inf')

        active_mask = (slots_tensor > 0)
        if active_mask.any():
            rows = torch.nonzero(active_mask).squeeze(1)
            mask[rows[:, None], self.mask_ban_eos] = -float('inf')

            len_check = (current_lengths + 1 + slots_tensor >= max_len)
            force_term_mask = active_mask & len_check
            if force_term_mask.any():
                rows_term = torch.nonzero(force_term_mask).squeeze(1)
                mask[rows_term[:, None], self.mask_force_terminal] = -float('inf')

        return mask

    def evaluate_smart(self, token_seq, X_padded_tensor, y_raw_tensor, coarse=True):
        seq_tuple = tuple(token_seq.tolist())
        if seq_tuple in self.expression_cache:
            return self.expression_cache[seq_tuple]

        if X_padded_tensor.dim() == 2:
            X_in = X_padded_tensor.unsqueeze(0)
        else:
            X_in = X_padded_tensor

        original_restarts = self.cfg.bfgs.n_restarts
        if coarse:
            self.cfg.bfgs.n_restarts = 1

        loss_val = 1e9
        sympy_expr = None

        try:
            pred_str_sub, _, loss, _ = bfgs.bfgs(
                list(seq_tuple), X_in, y_raw_tensor, self.cfg, self.test_data
            )
            if loss is not None:
                if isinstance(loss, complex) or np.iscomplexobj(loss):
                    loss = float(abs(loss)) + 1e6
                if np.isnan(loss) or np.isinf(loss):
                    loss = 1e9
                else:
                    loss = float(loss)
            else:
                loss = 1e9

            if isinstance(pred_str_sub, str):
                try:
                    sympy_expr = sp.sympify(pred_str_sub, evaluate=False)
                except:
                    sympy_expr = None
                    loss = 1e9

            if sympy_expr is not None:
                loss_val = loss

        except Exception as e:
            loss_val = 1e9
            sympy_expr = None

        finally:
            if coarse:
                self.cfg.bfgs.n_restarts = original_restarts

        result = (loss_val, sympy_expr)
        self.expression_cache[seq_tuple] = result
        return result

    def compute_full_metrics(self, sympy_expr, X_full_np, y_full_np, total_vars=10):
        try:
            all_vars = [sp.Symbol(f'x_{i + 1}') for i in range(total_vars)]
            f_numpy = sp.lambdify(all_vars, sympy_expr, modules=SAFE_MODULES)

            if X_full_np.ndim == 3: X_full_np = X_full_np[0]
            args = [X_full_np[:, i] for i in range(total_vars)]
            y_pred = f_numpy(*args)

            if np.ndim(y_pred) == 0: y_pred = np.full_like(y_full_np, float(y_pred))
            if np.iscomplexobj(y_pred): y_pred = np.real(y_pred)
            y_pred = y_pred.astype(np.float32)

            mask = np.isfinite(y_pred)
            if mask.sum() < len(y_pred) * 0.5: return float('inf'), -1.0

            mse = np.mean((y_full_np - y_pred) ** 2)
            r2 = r2_score(y_full_np, y_pred)
            return mse, r2
        except Exception:
            return float('inf'), -1.0

    def compute_aggressive_reward(self, loss, y_target_tensor):
        """
        激进型奖励函数：-log10(NMSE)
        """
        var_y = torch.var(y_target_tensor) + 1e-8
        loss_tensor = torch.tensor(loss, device=self.device, dtype=torch.float32)
        nmse = loss_tensor / var_y

        if nmse > 1.0:
            # R2 < 0
            return torch.max(torch.tensor(0.0), 1.0 - nmse * 0.1)
        else:
            # R2 > 0 -> Log Precision
            return -torch.log10(torch.clamp(nmse, min=1e-10))

    def run_collaborative(self, X_padded, y_target, n_iterations=50, batch_size=64, lr=0.01):
        """
        协同残差生成策略 (Final Edition Optimized)
        """
        print(f"[*] Starting Collaborative Generation...")
        print(f"    Config: Iter={n_iterations}, Batch={batch_size}, LR={lr}")

        y_bfgs = y_target.squeeze()
        X_padded_np = X_padded.squeeze(0).cpu().numpy()
        y_target_np = y_target.squeeze().cpu().numpy()

        # 预计算 Target 方差用于 Reward
        target_var = torch.var(y_bfgs) + 1e-8

        # 侦测未使用的变量
        unused_ids_list = self._detect_unused_variables(X_padded)
        unused_ids_tensor = torch.tensor(unused_ids_list, device=self.device, dtype=torch.long)
        if len(unused_ids_list) > 0:
            print(f"    [Masking] Auto-banned {len(unused_ids_list)} unused variables (Ids: {unused_ids_list})")

        # ---------------- 1. 初始化 ----------------
        with torch.no_grad():
            y_in = y_target.unsqueeze(2)
            encoder_input = torch.cat((X_padded, y_in), dim=2)
            encoder_input_enc = self.model.ieee_tran(encoder_input)
            points_emb_ = self.model.MultiModalEncoder.fc_points_(encoder_input_enc)
            points_emb = self.model.MultiModalEncoder.fc_points(encoder_input_enc)
            z_fixed = self.model.MultiModalEncoder.points_encoder(points_emb)
            pred_logits = self.model.MultiModalEncoder.token_predictor(points_emb_)
            # pred_indices = torch.argmax(pred_logits, dim=-1)
            _, pred_indices = torch.topk(pred_logits, k=self.top_k, dim=-1)
            v_curr = self.model.MultiModalEncoder.vq_layer.get_codebook_entry(pred_indices)

        # ---------------- 2. 获取 Visymre Baseline ----------------
        print("    [Init] Running Visymre fitfunc2 (Beam Search Baseline)...")
        try:
            output = self.model.fitfunc2(
                X_padded.squeeze(0),
                y_target.squeeze(0),
                cfg_params=self.cfg,
                test_data=self.test_data
            )
            best_expr_str = output['best_bfgs_preds'][0]
            visymre_expr = sp.sympify(best_expr_str)
            _, visymre_r2 = self.compute_full_metrics(visymre_expr, X_padded_np, y_target_np)
            print(f"    [Init] Visymre Baseline R2: {visymre_r2:.5f} | Expr: {visymre_expr}")
        except Exception as e:
            print(f"    [Warning] Visymre fitfunc2 failed: {e}")
            visymre_r2 = -float('inf')
            visymre_expr = None

        best_global_r2 = visymre_r2
        best_global_expr = visymre_expr

        # ---------------- 3. 视觉热启动 ----------------
        v_batch = v_curr.repeat(batch_size, 1, 1)  # Default

        if visymre_expr is not None:
            print("    [Init] �� Visual Warm Start Triggered! Rendering Teacher's Best Expr...")
            try:
                teacher_img = self.renderer.render(visymre_expr, X_padded_np, [0, 1]).to(self.device)
                with torch.no_grad():
                    visual_raw = self.model.MultiModalEncoder.visual_encoder(teacher_img)
                    quantized, _, _, _ = self.model.MultiModalEncoder.vq_layer(visual_raw)
                    v_curr = quantized.detach()
                    v_batch = v_curr.repeat(batch_size, 1, 1)
                print("    [Init] ✅ v_batch updated with Teacher's holographic features.")
            except Exception as e:
                print(f"    [Init] ⚠️ Visual Warm Start failed: {e}")

        # ---------------- 4. 初始化 Student ----------------
        if hasattr(self.model.fc_out, 'out_features'):
            vocab_size = self.model.fc_out.out_features
        else:
            vocab_size = len(self.id2word)

        max_len = 30
        student = LSTMStudent(vocab_size, embed_dim=64, hidden_dim=128).to(self.device)
        optimizer = optim.Adam(student.parameters(), lr=lr)

        z_batch = z_fixed.repeat(batch_size, 1, 1)
        top_k_percent = 0.05
        top_k_count = max(1, int(batch_size * top_k_percent))

        # [已移除] 之前的冗余定义 mask_force_eos ...

        # ---------------- 5. 迭代循环 ----------------
        for it in range(n_iterations):
            optimizer.zero_grad()

            progress = it / n_iterations
            student_alpha = 0.1 + 0.8 * progress

            with torch.no_grad():
                fused_enc = self.model.MultiModalEncoder.fusion_module_teacher(z_batch, v_batch)
                enc_src = fused_enc.permute(1, 0, 2)

            curr_tokens = torch.full((batch_size, 1), self.sos_idx, dtype=torch.long, device=self.device)
            finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
            stack_counters = torch.ones(batch_size, dtype=torch.long, device=self.device)

            log_probs_sum = torch.zeros(batch_size, device=self.device)
            entropies_sum = torch.zeros(batch_size, device=self.device)

            for t in range(max_len):
                with torch.no_grad():
                    seq_len = curr_tokens.size(1)
                    pos = self.model.pos_embedding(
                        torch.arange(0, seq_len, device=self.device).unsqueeze(0).repeat(batch_size, 1))
                    tok = self.model.tok_embedding(curr_tokens)
                    trg_emb = tok + pos
                    trg_mask1, trg_mask2 = self.model.make_trg_mask(curr_tokens)
                    output = self.model.decoder_transfomer(
                        trg_emb.permute(1, 0, 2), enc_src,
                        trg_mask2.bool(), tgt_key_padding_mask=trg_mask1.bool()
                    )
                    teacher_logits = self.model.fc_out(output.permute(1, 0, 2)[:, -1, :])

                student_logits = student(curr_tokens)

                # [关键] Teacher 温度降温 (Temperature Scaling)
                teacher_temp = 2.0
                soft_teacher_logits = teacher_logits / teacher_temp
                mixed_logits = (1 - student_alpha) * soft_teacher_logits + student_alpha * student_logits

                # [Prefix Syntax Masking]
                prefix_mask = self._get_prefix_mask(curr_tokens, batch_size, vocab_size, max_len,
                                                    ban_ids=unused_ids_tensor)
                mixed_logits = mixed_logits + prefix_mask

                dist = Categorical(logits=mixed_logits)
                next_token = dist.sample()
                greedy_token = torch.argmax(mixed_logits, dim=-1)
                next_token[0] = greedy_token[0]

                # Update Stack
                arity_delta = torch.zeros(batch_size, dtype=torch.long, device=self.device)
                token_vals = next_token.cpu().numpy()
                for b_idx, tid in enumerate(token_vals):
                    if tid in self.arity_2_set:
                        arity_delta[b_idx] = 1
                    elif tid in self.arity_1_set:
                        arity_delta[b_idx] = 0
                    elif tid in self.arity_0_set:
                        arity_delta[b_idx] = -1

                stack_counters = stack_counters + arity_delta * (~finished).long()

                log_prob = dist.log_prob(next_token)
                entropy = dist.entropy()
                mask = (~finished).float()
                log_probs_sum = log_probs_sum + log_prob * mask
                entropies_sum = entropies_sum + entropy * mask

                curr_tokens = torch.cat([curr_tokens, next_token.unsqueeze(1)], dim=1)
                finished = finished | (stack_counters == 0)
                if finished.all(): break

            # --- 评估 ---
            rewards = []
            results = []
            iter_best_loss = float('inf')
            iter_best_expr = None
            greedy_loss = float('inf')

            for i in range(batch_size):
                loss, expr = self.evaluate_smart(curr_tokens[i], X_padded, y_bfgs, coarse=True)

                # [激进奖励] -log10(NMSE)
                r = self.compute_aggressive_reward(loss, y_bfgs)

                rewards.append(r.item())
                results.append((loss, i))

                if loss < iter_best_loss:
                    iter_best_loss = loss
                    iter_best_expr = expr

                if i == 0: greedy_loss = loss

            iter_best_r2 = -float('inf')
            if iter_best_expr is not None and iter_best_loss < 5.0:
                _, iter_best_r2 = self.compute_full_metrics(iter_best_expr, X_padded_np, y_target_np)
                if iter_best_r2 > best_global_r2:
                    best_global_r2 = iter_best_r2
                    best_global_expr = iter_best_expr

            print(
                f"  [Iter {it:02d}] S_Alpha={student_alpha:.2f} | B_MSE={iter_best_loss:.4f} | IterR2={iter_best_r2:.4f} | GlobalR2={best_global_r2:.4f}")

            if best_global_r2 > 0.999:
                print(f"\n[!] Target Accuracy Reached. Early Stopping.")
                return best_global_expr

            # --- 优化 (Top-k) ---
            results.sort(key=lambda x: x[0])
            top_indices = [idx for (loss, idx) in results[:top_k_count]]
            train_mask = torch.zeros(batch_size, device=self.device)
            train_mask[top_indices] = 1.0

            rewards_t = torch.tensor(rewards, device=self.device)
            selected_rewards = rewards_t[top_indices]

            if len(selected_rewards) > 1 and selected_rewards.std() > 1e-6:
                baseline = selected_rewards.mean()
                adv = (rewards_t - baseline)
            else:
                adv = rewards_t

            avg_len = (curr_tokens != self.pad_idx).sum(dim=1).float().mean()
            len_penalty = 0.001 * avg_len

            pg_loss = -(log_probs_sum * adv * train_mask).sum() / (train_mask.sum() + 1e-6)
            entropy_loss = -entropies_sum.mean()

            total_loss = pg_loss + 0.005 * entropy_loss + len_penalty
            total_loss.backward()
            optimizer.step()

            # --- 全息闭环 ---
            beat_teacher = (iter_best_r2 > visymre_r2)
            is_new_best = (iter_best_r2 >= best_global_r2)

            if iter_best_expr is not None and iter_best_loss < 100.0 and beat_teacher and is_new_best:
                print(
                    f"    >>> Feedback Triggered! Updating Visual Context (R2: {iter_best_r2:.4f} > Teacher: {visymre_r2:.4f})")
                try:
                    hrs_img = self.renderer.render(iter_best_expr, X_padded_np, [0, 1]).to(self.device)
                    with torch.no_grad():
                        visual_raw = self.model.MultiModalEncoder.visual_encoder(hrs_img)
                        quantized, _, _, _ = self.model.MultiModalEncoder.vq_layer(visual_raw)
                        v_batch = quantized.detach().repeat(batch_size, 1, 1)
                except:
                    pass

        print(f"\n[*] Final Best: R2={best_global_r2:.5f}")
        return best_global_expr