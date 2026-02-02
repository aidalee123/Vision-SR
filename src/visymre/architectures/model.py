import torch
from concurrent.futures import ProcessPoolExecutor, as_completed
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from .MultimodalEncoder import MultiModalEncoder
from .beam_search import BeamHypotheses
import numpy as np
from .bfgs import bfgs
import math
from torch.optim.lr_scheduler import LambdaLR

def bfgs_wrapper(args):
    ww, X_cpu, y_cpu, cfg_params, test_data = args
    try:
        pred_w_c, constants, loss_bfgs, exa = bfgs(ww, X_cpu, y_cpu, cfg_params, test_data)
        return (str(pred_w_c), loss_bfgs, ww)
    except Exception:
        return (None, float("nan"), ww)


class Model(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.automatic_optimization = False
        self.cfg = cfg
        self.MultiModalEncoder = MultiModalEncoder(cfg)

        self.trg_pad_idx = cfg.trg_pad_idx
        self.tok_embedding = nn.Embedding(cfg.output_dim, cfg.dim_hidden)
        self.pos_embedding = nn.Embedding(cfg.length_eq, cfg.dim_hidden)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=cfg.dim_hidden,
            activation='gelu',
            nhead=cfg.num_heads,
            dim_feedforward=2 * cfg.dim_hidden,
            norm_first=True
        )
        self.decoder_transfomer = nn.TransformerDecoder(decoder_layer,num_layers=cfg.dec_layers)
        self.fc_out = nn.Linear(cfg.dim_hidden, cfg.output_dim)

        # Loss Functions
        self.CrossEntropy_Loss = nn.CrossEntropyLoss(ignore_index=0)
        self.MSE_Loss = nn.MSELoss()  # For generation supervision

        self.num_epochs = cfg.epochs
        self.lambda_gen = 1.0  # Weight for generation loss


    def make_trg_mask(self, trg):
        trg_pad_mask = (trg != self.trg_pad_idx).float()
        trg_pad_mask = (
            trg_pad_mask.masked_fill(trg_pad_mask == 0, float("-inf"))
            .masked_fill(trg_pad_mask == 1, float(0.0))
            .type_as(trg)
        )
        trg_len = trg.shape[1]
        mask = (torch.triu(torch.ones(trg_len, trg_len)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
            .type_as(trg)
        )
        return trg_pad_mask, mask

    def decoder_output(self, trg_, encoder_output, trg_mask1, trg_mask2):
        # Note: encoder_output is [B, 1, D], we need to permute for transformer [L, B, D]
        output = self.decoder_transfomer(
            trg_.permute(1, 0, 2),
            encoder_output.permute(1, 0, 2),
            trg_mask2.bool(),
            tgt_key_padding_mask=trg_mask1.bool(),
        )
        return output

    def float2bit(self, f, num_e_bits=8, num_m_bits=8, bias=127., dtype=torch.float32):
        s = (torch.sign(f + 0.001) * -1 + 1) * 0.5
        s = s.unsqueeze(-1)
        f1 = torch.abs(f)
        e_scientific = torch.floor(torch.log2(f1))
        e_scientific[e_scientific == float("-inf")] = -(2 ** (num_e_bits - 1) - 1)
        e_decimal = e_scientific + (2 ** (num_e_bits - 1) - 1)
        e = self.integer2bit(e_decimal, num_bits=num_e_bits)
        f2 = f1 / 2 ** (e_scientific)
        m2 = self.remainder2bit(f2 % 1, num_bits=int(bias))
        fin_m = m2[:, :, :, :num_m_bits]
        return torch.cat([s, e, fin_m], dim=-1).type(dtype)

    def remainder2bit(self, remainder, num_bits=127):
        dtype = remainder.type()
        exponent_bits = torch.arange(num_bits, device=self.device).type(dtype)
        exponent_bits = exponent_bits.repeat(remainder.shape + (1,))
        out = (remainder.unsqueeze(-1) * 2 ** exponent_bits) % 1
        return torch.floor(2 * out)

    def integer2bit(self, integer, num_bits=8):
        dtype = integer.type()
        exponent_bits = -torch.arange(-(num_bits - 1), 1, device=self.device).type(dtype)
        exponent_bits = exponent_bits.repeat(integer.shape + (1,))
        out = integer.unsqueeze(-1) / 2 ** exponent_bits
        return (out - (out % 1)) % 2

    def ieee_tran(self, input):
        input = self.float2bit(input)
        input = input.view(input.shape[0], input.shape[1], -1)
        input = (input - 0.5) * 2
        return input

    def forward(self, batch, num_epochs=None, current_epoch=None,batch_idx=0):
        # Prepare Inputs
        b = batch[0].permute(0, 2, 1)
        size = b.shape[-1]
        src_x = b[:, :, :(size - 1)]
        src_y = b[:, :, -1].unsqueeze(2)

        # Prepare Target & Masks
        trg = batch[1].long()
        pos = self.pos_embedding(
            torch.arange(0, batch[1].shape[1] - 1)
            .unsqueeze(0)
            .repeat(batch[1].shape[0], 1)
            .type_as(trg)
        )
        te = self.tok_embedding(trg[:, :-1])
        trg_ = te + pos
        trg_mask1, trg_mask2 = self.make_trg_mask(trg[:, :-1])

        # Prepare Dataset Features (IEEE encoding)
        encoder_input = torch.cat((src_x, src_y), dim=-1)
        encoder_input = self.ieee_tran(encoder_input)

        funimage_gt = batch[2].permute(0, 3, 1, 2)  # [B, 3, H, W]

        # --- MultiModal Forward ---
        # Returns fused features and reconstructed image
        fused_out_student, fused_out_teacher, aux_losses = self.MultiModalEncoder.forward(
            encoder_input,
            gt_image=funimage_gt,

        )
        if fused_out_teacher is not  None:
        # --- Symbolic Decoding ---
            output_logits_student = self.fc_out(self.decoder_output(trg_, fused_out_student, trg_mask1, trg_mask2))
            output_logits_teacher = self.fc_out(self.decoder_output(trg_, fused_out_teacher, trg_mask1, trg_mask2))
        else:
            output_logits_student = self.fc_out(self.decoder_output(trg_, fused_out_student, trg_mask1, trg_mask2))
            output_logits_teacher =None

        return output_logits_student, output_logits_teacher, aux_losses, trg

    def calculate_kd_loss(self,student_logits, teacher_logits, temperature=1.0):
        student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
        with torch.no_grad():
            teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
        loss_kd = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
        loss_kd = loss_kd * (temperature ** 2)

        return loss_kd
    def compute_loss(self, output_logits_student, output_logits_teacher, aux_losses, trg):
        # 1. Symbolic Regression Loss
        trg_flat = trg[:, 1:].contiguous().view(-1)
        output_flat_student = output_logits_student.permute(1, 0, 2).contiguous().view(-1, output_logits_student.shape[-1])
        loss_sr_student = self.CrossEntropy_Loss(output_flat_student, trg_flat)

        # 2. VQ Auxiliary Losses
        loss_vq = aux_losses.get('vq_loss', 0.0)
        loss_token = aux_losses.get('token_loss', 0.0)
        acc_top1 = aux_losses.get('acc_top1', 0.0)
        acc_top5 = aux_losses.get('acc_top5', 0.0)
        sentinel_usage = aux_losses.get('sentinel_usage', 0.0)
        contrastive_loss = aux_losses.get('contrastive_loss', 0.0)
        acc_top3 = aux_losses.get('acc_top3', 0.0)
        noise_ratio_bias = aux_losses.get('noise_ratio_bias', 0.0)
        if output_logits_teacher is not None:
            output_flat_teacher = output_logits_teacher.permute(1, 0, 2).contiguous().view(-1,
                                                                                           output_logits_teacher.shape[
                                                                                               -1])
            loss_sr_teacher = self.CrossEntropy_Loss(output_flat_teacher, trg_flat)
            kd_loss = self.calculate_kd_loss(output_flat_student, output_flat_teacher, temperature=1.0)
        else:
            loss_sr_teacher = torch.tensor(0)
            kd_loss = torch.tensor(0)
        total_loss = loss_sr_student + loss_sr_teacher + 0.1 * loss_vq + 0.1111 * loss_token+0.1*kd_loss + 0.1*contrastive_loss

        return (total_loss, loss_sr_student , loss_sr_teacher, loss_vq,
                loss_token, acc_top1,acc_top5, sentinel_usage,contrastive_loss,acc_top3,noise_ratio_bias)

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            train_dataset_size = len(self.trainer.datamodule.train_dataloader().dataset)
            batch_size = self.cfg.batch_size
            self.steps_per_epoch = math.ceil(train_dataset_size / batch_size)
            self.total_steps = self.steps_per_epoch * self.cfg.epochs

    def on_train_epoch_start(self):
        if self.current_epoch == 30:
            self.freeze_visual_modules()
    def freeze_visual_modules(self):
        modules_to_freeze = [
            self.MultiModalEncoder.visual_encoder,
            self.MultiModalEncoder.vq_layer
        ]
        for module in modules_to_freeze:
            # 切换到评估模式 (对 ResNet 的 BN 层至关重要)
            module.eval()
            # 关闭参数梯度
            for param in module.parameters():
                param.requires_grad = False

        print("INFO: MultiModalEncoder.visual_encoder and vq_layer representational weights are FROZEN.")
    def training_step(self, batch, batch_idx):
        optimizer = self.optimizers()
        scheduler = self.lr_schedulers()
        optimizer.zero_grad()

        # Forward Pass (End-to-End)
        output_logits_student, output_logits_teacher, aux_losses, trg = self.forward(
            batch,
            num_epochs=self.num_epochs,
            current_epoch=self.current_epoch,
            batch_idx=batch_idx
        )
        # Compute Loss
        (loss, loss_sr_student , loss_sr_teacher, loss_vq, loss_token,
         acc_top1,acc_top5,sentinel_usage,contrastive_loss,acc_top3,noise_ratio_bias)= self.compute_loss(output_logits_student, output_logits_teacher, aux_losses, trg)

        self.log("train_loss", loss, on_step=True, on_epoch=True)
        self.log("loss_sr_student", loss_sr_student, on_step=True)
        if loss_sr_teacher is not  None:
            self.log("loss_sr_teacher", loss_sr_teacher, on_step=True)
        self.log("loss_vq", loss_vq, on_step=True)
        self.log("loss_token", loss_token, on_step=True)
        self.log("contrastive_loss", contrastive_loss, on_step=True)

        if 'perplexity' in aux_losses:
            self.log("codebook_perplexity", aux_losses['perplexity'], on_step=True, on_epoch=True, prog_bar=True)
        self.manual_backward(loss)
        if self.current_epoch >=30:
            if hasattr(self, 'MultiModalEncoder'):
                self.MultiModalEncoder.visual_encoder.eval()
                self.MultiModalEncoder.vq_layer.eval()
        # torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler:
            scheduler.step()

    def validation_step(self, batch, _):
        output_logits_student, output_logits_teacher, aux_losses, trg = self.forward(
            batch,
            num_epochs=self.num_epochs,
            current_epoch=self.current_epoch
        )
        # Compute Loss
        loss, loss_sr_student , loss_sr_teacher, loss_vq, loss_token, acc_top1,acc_top5,sentinel_usage, contrastive_loss,acc_top3,noise_ratio_bias= (
            self.compute_loss(output_logits_student, output_logits_teacher, aux_losses, trg))
        self.log("val_loss", loss_sr_student, on_epoch=True)
        if loss_sr_teacher is not  None:
            self.log("val_loss_teacher", loss_sr_teacher, on_epoch=True)
    def get_lr_lambda(self, current_step):
        total_steps = getattr(self, "total_steps", 1)
        progress = float(current_step) / float(max(1, total_steps))
        lr_mult = 1.0 - 0.9 * (1.0 - math.cos(math.pi * 0.5 * progress))
        return lr_mult

    def configure_optimizers(self):
        decay_params = []
        no_decay_params = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if any(nd in name for nd in ["bias", "LayerNorm.weight"]):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optimizer = torch.optim.AdamW(
            [
                {"params": decay_params, "weight_decay": 1e-3},
                {"params": no_decay_params, "weight_decay": 0.0}
            ],
            lr=self.cfg.lr
        )
        scheduler = {
            'scheduler': LambdaLR(optimizer, self.get_lr_lambda),
            'interval': 'step',
            'frequency': 1,
        }
        return [optimizer], [scheduler]

    def fitfunc2(self, X, y, cfg_params=None, test_data=None):
        print("beam = ", cfg_params.beam_size)
        w2i = test_data.word2id
        unary_names = ['abs', 'asin', 'cos', 'exp', 'ln', 'sin', 'sqrt', 'tan']
        binary_names = ['add', 'div', 'mul', 'pow', 'sub']
        #transcendental_names = [ 'cos', 'exp', 'ln', 'sin', 'tan']
        transcendental_names = []
        arity_1_ids = {w2i[name] for name in unary_names if name in w2i}
        arity_2_ids = {w2i[name] for name in binary_names if name in w2i}
        all_op_ids = arity_1_ids | arity_2_ids

        transcendental_ids = {w2i[name] for name in transcendental_names if name in w2i}
        pow_id = w2i.get('pow')

        limit_pow_const = getattr(cfg_params, 'no_c_in_pow', False)
        if limit_pow_const:
            c_id = w2i.get('c', 3)
            print("Info: Constraint [No Constant in Pow] is ENABLED.")
        else:
            c_id = None

        pad_id = w2i.get('P', 0)
        start_id = w2i.get('S', 1)
        finish_id = w2i.get('F', 2)

        X = X
        y = y[:, None]
        X = X.clone().detach().to(cfg_params.device).unsqueeze(0)
        if X.shape[2] < self.cfg.dim_input - 1:
            pad = torch.zeros(1, X.shape[1], self.cfg.dim_input - X.shape[2] - 1, device=cfg_params.device)
            X = torch.cat((X, pad), dim=2)

        input_X = X[0, :, :10]
        abs_sum = torch.abs(input_X).sum(dim=0)
        unused_feat_indices = (abs_sum == 0).nonzero(as_tuple=True)[0].cpu().numpy()
        masked_var_ids = set()
        for idx in unused_feat_indices:
            var_name = f"x_{idx + 1}"
            if var_name in w2i:
                masked_var_ids.add(w2i[var_name])
            elif f"x_{idx}" in w2i:
                masked_var_ids.add(w2i[f"x_{idx}"])

        y = y.clone().detach().to(cfg_params.device).unsqueeze(0)

        with torch.no_grad():
            # 随机采样
            n_points = X.shape[1]
            if n_points > 200:
                indices = torch.randperm(n_points, device=cfg_params.device)[:200]
                indices, _ = torch.sort(indices)
                X_enc = X[:, indices, :]
                y_enc = y[:, indices, :]
                encoder_input = torch.cat((X_enc, y_enc), dim=2)
            else:
                encoder_input = torch.cat((X, y), dim=2)

            encoder_input = self.ieee_tran(encoder_input)
            src_enc = self.MultiModalEncoder.predict(encoder_input)

            shape_enc_src = (cfg_params.beam_size,) + src_enc.shape[1:]
            enc_src = src_enc.unsqueeze(1).expand((1, cfg_params.beam_size) + src_enc.shape[1:]).contiguous().view(
                shape_enc_src)

            generated = torch.zeros([cfg_params.beam_size, self.cfg.length_eq], dtype=torch.long,
                                    device=cfg_params.device)
            generated[:, 0] = start_id

            generated_hyps = BeamHypotheses(cfg_params.beam_size, self.cfg.length_eq, 1.0, 1)
            done = False
            beam_scores = torch.zeros(cfg_params.beam_size, device=cfg_params.device, dtype=torch.float)
            beam_scores[1:] = -1e9

            cur_len = torch.tensor(1, device=cfg_params.device, dtype=torch.int64)
            cache = {"slen": 0}

            while cur_len < self.cfg.length_eq:
                generated_mask1, generated_mask2 = self.make_trg_mask(generated[:, :cur_len])
                pos = self.pos_embedding(
                    torch.arange(0, cur_len).unsqueeze(0).repeat(generated.shape[0], 1).type_as(generated))
                te = self.tok_embedding(generated[:, :cur_len])
                trg_ = te + pos
                output = self.decoder_transfomer(
                    trg_.permute(1, 0, 2), enc_src.permute(1, 0, 2),
                    generated_mask2.float(), tgt_key_padding_mask=generated_mask1.bool()
                )
                output = self.fc_out(output).permute(1, 0, 2).contiguous()
                scores = F.log_softmax(output[:, -1:, :], dim=-1).squeeze(1)
                n_words = scores.shape[-1]

                # --- Constraint Logic ---
                logit_mask = torch.zeros_like(scores)

                for i in range(cfg_params.beam_size):
                    if beam_scores[i] < -1e8: continue

                    hyp_seq = generated[i, :cur_len].cpu().tolist()

                    valency, forbidden_by_structure = self._analyze_prefix_tree_context(
                        hyp_seq, arity_1_ids, arity_2_ids,
                        transcendental_ids, pow_id, c_id, start_id
                    )

                    forbidden = set()
                    forbidden.update(forbidden_by_structure)

                    remaining_len = self.cfg.length_eq - cur_len.item()
                    if valency >= remaining_len:
                        forbidden.update(all_op_ids)

                    if valency > 0:
                        forbidden.add(finish_id)
                        forbidden.add(pad_id)

                    forbidden.update(masked_var_ids)

                    if forbidden:
                        valid_forbidden = [x for x in forbidden if x < n_words]
                        if valid_forbidden:
                            logit_mask[i, valid_forbidden] = float('-inf')

                scores = scores + logit_mask
                # ------------------------

                _scores = scores + beam_scores[:, None].expand_as(scores)
                _scores = _scores.view(cfg_params.beam_size * n_words)
                next_scores, next_words = torch.topk(_scores, 2 * cfg_params.beam_size, dim=0, largest=True,
                                                     sorted=True)

                done = done or generated_hyps.is_done(next_scores.max().item())
                next_sent_beam = []
                for idx, value in zip(next_words, next_scores):
                    beam_id = idx // n_words
                    word_id = idx % n_words
                    if word_id == finish_id:
                        generated_hyps.add(generated[beam_id, :cur_len].clone().cpu(), value.item())
                    else:
                        next_sent_beam.append((value, word_id, beam_id))
                    if len(next_sent_beam) == cfg_params.beam_size: break

                if len(next_sent_beam) == 0:
                    next_sent_beam = [(0, self.trg_pad_idx, 0)] * cfg_params.beam_size

                beam_scores = torch.tensor([x[0] for x in next_sent_beam], device=cfg_params.device)
                beam_words = torch.tensor([x[1] for x in next_sent_beam], device=cfg_params.device)
                beam_idx = torch.tensor([x[2] for x in next_sent_beam], device=cfg_params.device)
                generated = generated[beam_idx, :]
                generated[:, cur_len] = beam_words
                for k in cache.keys():
                    if k != "slen": cache[k] = (cache[k][0][beam_idx], cache[k][1][beam_idx])
                cur_len = cur_len + 1

            # ... (BFGS Parallel Part) ...
            best_preds_bfgs = []
            best_L_bfgs = []
            best_token = []
            token = []
            L_bfgs = []
            P_bfgs = []

            if 3 in test_data.id2word:
                test_data.id2word[3] = "constant"
            elif 'c' in w2i:
                test_data.id2word[w2i['c']] = "constant"

            X_cpu = X.cpu()
            y_cpu = y.cpu()
            sorted_hyps = sorted(generated_hyps.hyp, key=lambda x: x[0], reverse=True)

            valid_hyps = []
            for score, ww in sorted_hyps:
                # 1. 确保 ww 转为 Python List (处理 Tensor 或 Numpy)
                if isinstance(ww, torch.Tensor):
                    seq = ww.cpu().tolist()
                elif isinstance(ww, np.ndarray):
                    seq = ww.tolist()
                else:
                    seq = list(ww)

                if finish_id in seq:
                    seq = seq[:seq.index(finish_id)]

                seq = [s for s in seq if s != pad_id]
                valency, _ = self._analyze_prefix_tree_context(
                    seq, arity_1_ids, arity_2_ids,
                    transcendental_ids, pow_id, c_id, start_id
                )

                if valency == 0:
                    valid_hyps.append((score, ww))

            task_list = [(ww, X_cpu, y_cpu, cfg_params, test_data) for _, ww in valid_hyps]

            if len(task_list) == 0 and len(sorted_hyps) > 0:
                print("Warning: All beams were filtered out due to invalid structure. Fallback to top-1 raw.")
                task_list = [(sorted_hyps[0][1], X_cpu, y_cpu, cfg_params, test_data)]

            if len(task_list) > 0:
                with ProcessPoolExecutor(20) as executor:
                    futures = [executor.submit(bfgs_wrapper, args) for args in task_list]
                    for future in as_completed(futures):
                        res = future.result()
                        pred_w_c, loss_bfgs, ww = res
                        if pred_w_c is not None:
                            P_bfgs.append(pred_w_c)
                            L_bfgs.append(loss_bfgs)
                            token.append(ww)

            if len(L_bfgs) == 0 or all(np.isnan(np.array(L_bfgs))):
                L_bfgs = [float("nan")]
                best_L_bfgs = [float("nan")]
                best_preds_bfgs = [None]
                best_token = [None]
            else:
                best_idx = np.nanargmin(L_bfgs)
                best_preds_bfgs.append(P_bfgs[best_idx])
                best_L_bfgs.append(L_bfgs[best_idx])
                best_token.append(token[best_idx])

            output = {
                'pred_target': generated_hyps.hyp[0][1] if generated_hyps.hyp else [],
                'all_bfgs_preds': P_bfgs,
                'all_bfgs_loss': L_bfgs,
                'best_bfgs_preds': best_preds_bfgs,
                'best_bfgs_loss': best_L_bfgs,
                'best_token': best_token
            }
            self.eq = output['best_bfgs_preds']
            return output

    def _analyze_prefix_tree_context(self, seq, arity_1_ids, arity_2_ids, transcendental_ids, pow_id, c_id, start_id=1):

        stack = [[None, 1, set()]]

        start_idx = 0
        if len(seq) > 0 and seq[0] == start_id:
            start_idx = 1

        for token in seq[start_idx:]:
            if len(stack) == 0: break
            stack[-1][1] -= 1
            current_inherited_constraints = stack[-1][2].copy()

            if c_id is not None:
                if len(stack) > 0 and stack[-1][0] == pow_id and stack[-1][1] == 0:
                    current_inherited_constraints.add(c_id)

            new_constraints_for_children = current_inherited_constraints.copy()

            if token in transcendental_ids:
                new_constraints_for_children.update(transcendental_ids)

            if pow_id is not None and token == pow_id:
                new_constraints_for_children.add(pow_id)
            if token in arity_2_ids:
                stack.append([token, 2, new_constraints_for_children])
            elif token in arity_1_ids:
                stack.append([token, 1, new_constraints_for_children])

            while len(stack) > 0 and stack[-1][1] == 0:
                stack.pop()

        valency = sum([s[1] for s in stack])
        current_forbidden_set = stack[-1][2].copy() if len(stack) > 0 else set()
        if c_id is not None:
            if len(stack) > 0 and stack[-1][0] == pow_id and stack[-1][1] == 1:
                current_forbidden_set.add(c_id)

        return valency, current_forbidden_set


