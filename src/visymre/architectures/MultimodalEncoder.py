
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchvision.models as models

class ISAB(nn.Module):
    """
    Induced Set Attention Block (ISAB) - Pre-LayerNorm Version.
    Consistent with norm_first=True in the rest of the code.
    """

    def __init__(self, hidden_dim, num_heads, num_inds=64):
        super().__init__()
        # Inducing Points
        self.I = nn.Parameter(torch.Tensor(1, num_inds, hidden_dim))
        nn.init.xavier_uniform_(self.I)

        # --- MAB 1: Inducing points attend to Input X ---
        # Norms for Pre-LN
        self.norm1_I = nn.LayerNorm(hidden_dim)  # Norm before Q
        self.norm1_X = nn.LayerNorm(hidden_dim)  # Norm before K, V

        self.attn1 = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)

        self.norm2 = nn.LayerNorm(hidden_dim)  # Norm before FFN
        self.ff1 = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.Linear(2 * hidden_dim, hidden_dim)
        )

        # --- MAB 2: Input X attends to Inducing points H ---
        # Norms for Pre-LN
        self.norm3_X = nn.LayerNorm(hidden_dim)  # Norm before Q
        self.norm3_H = nn.LayerNorm(hidden_dim)  # Norm before K, V

        self.attn2 = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)

        self.norm4 = nn.LayerNorm(hidden_dim)  # Norm before FFN
        self.ff2 = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.Linear(2 * hidden_dim, hidden_dim)
        )

    def forward(self, x):
        """
        x: [B, N, D]
        """
        B = x.shape[0]
        I_expanded = self.I.expand(B, -1, -1)  # [B, M, D]

        # ===========================
        # Step 1: H = Attn(I, X)
        # ===========================
        # Pre-LN: Norm inputs -> Attn -> Add Residual

        # Q comes from I (Residual stream for Step 1)
        q1 = self.norm1_I(I_expanded)

        # K, V come from X
        k1 = v1 = self.norm1_X(x)

        h_attn, _ = self.attn1(query=q1, key=k1, value=v1)
        h = I_expanded + h_attn  # Residual connection with I

        # FFN with Pre-LN
        h = h + self.ff1(self.norm2(h))

        # ===========================
        # Step 2: Out = Attn(X, H)
        # ===========================
        # Pre-LN: Norm inputs -> Attn -> Add Residual

        # Q comes from X (Residual stream for Step 2)
        q2 = self.norm3_X(x)
        # K, V come from H
        k2 = v2 = self.norm3_H(h)

        out_attn, _ = self.attn2(query=q2, key=k2, value=v2)
        out = x + out_attn  # Residual connection with X

        # FFN with Pre-LN
        out = out + self.ff2(self.norm4(out))


        return out


class PointsEncoder(pl.LightningModule):
    def __init__(self, num_layers, hidden_dim, num_heads, num_inds=64):
        super().__init__()
        self.layers = nn.ModuleList([
            ISAB(hidden_dim, num_heads, num_inds=num_inds)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
class ResNetVisualEncoder(nn.Module):
    def __init__(self, output_dim=512, input_channels=3):
        super().__init__()
        resnet = models.resnet18(pretrained=False)  # weights=None
        original_conv1 = resnet.conv1
        resnet.conv1 = nn.Conv2d(
            in_channels=input_channels,
            out_channels=original_conv1.out_channels,
            kernel_size=3,
            stride=4,
            padding=original_conv1.padding,
            bias=True
        )
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        self.proj = nn.Linear(512, output_dim) if output_dim != 512 else nn.Identity()
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(2)
        x = x.permute(0, 2, 1)
        x = self.proj(x)
        x = self.norm(x)
        return x

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25, diversity_weight=0.1):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.diversity_weight = diversity_weight

        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.num_embeddings, 1.0 / self.num_embeddings)

    def forward(self, inputs):
        flat_input = inputs.contiguous().view(-1, self.embedding_dim)
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self.embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self.embedding.weight.t()))

        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        quantized = self.embedding(encoding_indices).view(inputs.shape)

        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss_vq = q_latent_loss + self.commitment_cost * e_latent_loss

        probs = F.softmax(-distances, dim=-1)
        avg_probs = torch.mean(probs, dim=0)
        loss_diversity = torch.sum(avg_probs * torch.log(avg_probs + 1e-10))
        total_loss = loss_vq + self.diversity_weight * loss_diversity

        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        quantized = inputs + (quantized - inputs).detach()

        return quantized, total_loss, encoding_indices.view(inputs.shape[0], inputs.shape[1]), perplexity

    def get_codebook_entry(self, indices):
        return self.embedding(indices)


class VirtualVisualDecoder(nn.Module):
    def __init__(self, hidden_dim, num_image_tokens, num_patches=49, nuum_layer=2, num_heads=8):
        super().__init__()
        self.num_patches = num_patches
        self.codebook_size = num_image_tokens
        self.query_embed = nn.Parameter(torch.randn(1, num_patches, hidden_dim))

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=nuum_layer)
        self.to_logits = nn.Linear(hidden_dim, num_image_tokens)

    def forward(self, dataset_features, image_indices=None):
        B = dataset_features.shape[0]
        queries = self.query_embed.expand(B, -1, -1)
        features = self.transformer(tgt=queries, memory=dataset_features)
        logits = self.to_logits(features)
        return logits


class CrossAttentionFusion(pl.LightningModule):

    def __init__(self, hidden_dim, num_heads, norm_inputs=True):
        super().__init__()
        self.norm_inputs = norm_inputs

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            bias=True,
            batch_first=True
        )

        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )

        self.norm_q = nn.LayerNorm(hidden_dim)
        self.norm_k = nn.LayerNorm(hidden_dim)
        self.norm_ff = nn.LayerNorm(hidden_dim)
        self.norm_out = nn.LayerNorm(hidden_dim)

    def forward(self, x, y):
        residual = x
        x_norm = self.norm_q(x)
        y_norm = self.norm_k(y)
        attn_out, attn_weights = self.attention(
            query=x_norm, key=y_norm, value=y_norm,
            need_weights=True
        )
        x = residual + attn_out
        residual = x
        x_norm = self.norm_ff(x)
        ff_out = self.ff(x_norm)
        out = residual + ff_out
        return out, None, attn_weights

class BiasCrossAttentionFusion(nn.Module):

    def __init__(self, hidden_dim, num_heads=8, bias_proj_dim=64,
                 vis_save_dir='./attention_vis'):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads
        self.vis_save_dir = vis_save_dir
        os.makedirs(self.vis_save_dir, exist_ok=True)
        assert self.head_dim * num_heads == self.hidden_dim, "hidden_dim error"
        self.w_q = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.w_k = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.w_v = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.w_out = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.geo_q_proj = nn.Linear(hidden_dim, bias_proj_dim, bias=True)
        self.geo_k_proj = nn.Linear(hidden_dim, bias_proj_dim, bias=True)
        self.pos_scale = nn.Parameter(torch.tensor(2.0))
        self.neg_scale = nn.Parameter(torch.tensor(50.0))
        self.attn_logit_scale = nn.Parameter(torch.log(torch.tensor(10.0)))
        self.norm_q = nn.LayerNorm(hidden_dim)
        self.norm_kv = nn.LayerNorm(hidden_dim)
        self.norm_out = nn.LayerNorm(hidden_dim)

        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim, bias=True),
            nn.GELU(),
            nn.Linear(2 * hidden_dim, hidden_dim, bias=True),
        )

    def forward(self, dataset_features, visual_features, current_step=None):
        B, N, D = dataset_features.shape
        M = visual_features.shape[1]
        residual = dataset_features
        q_in = self.norm_q(dataset_features)
        kv_in = self.norm_kv(visual_features)
        geo_q = self.geo_q_proj(q_in)
        geo_k = self.geo_k_proj(kv_in)
        bias_raw = torch.bmm(F.normalize(geo_q, p=2), F.normalize(geo_k, p=2).transpose(1, 2))
        bias_pos = F.relu(bias_raw) * self.pos_scale
        bias_neg = -F.relu(-bias_raw) * self.neg_scale
        full_bias = bias_pos + bias_neg
        q = self.w_q(q_in).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.w_k(kv_in).view(B, M, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.w_v(kv_in).view(B, M, self.num_heads, self.head_dim).transpose(1, 2)
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_logits = attn_logits * self.attn_logit_scale.exp()
        scores = attn_logits + full_bias.unsqueeze(1)
        attn_weights = F.softmax(scores, dim=-1)
        out = torch.matmul(attn_weights, v).transpose(1, 2).contiguous().view(B, N, D)
        out = self.w_out(out)
        out = residual + out
        residual = out
        out = self.norm_out(out)
        out = residual + self.ff(out)
        return out

class MultiModalEncoder(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.top_k = cfg.top_k
        self.hidden_dim = cfg.dim_hidden
        self.codebook_size = cfg.codebook_size
        self.num_visual_patches = cfg.num_patches

        self.fc_points = nn.Linear(cfg.points_dim_input, cfg.dim_hidden, bias=True)
        self.fc_points_ = nn.Linear(cfg.points_dim_input, cfg.dim_hidden, bias=True)

        self.points_encoder = PointsEncoder(
            cfg.n_l_points_encoder, cfg.dim_hidden, cfg.num_heads, cfg.num_inds
        )
        self.points_encoder_= PointsEncoder(
            1, cfg.dim_hidden, cfg.num_heads, cfg.num_inds
        )

        self.visual_encoder = ResNetVisualEncoder(output_dim=cfg.dim_hidden, input_channels=cfg.input_channels)

        self.vq_layer = VectorQuantizer(
            num_embeddings=self.codebook_size,
            commitment_cost=0.25,
            embedding_dim=cfg.dim_hidden,
            diversity_weight=0.001
        )

        self.token_predictor = VirtualVisualDecoder(
            hidden_dim=cfg.dim_hidden,
            num_heads=cfg.num_heads,
            num_image_tokens=self.codebook_size,
            num_patches=self.num_visual_patches,
            nuum_layer=cfg.n_l_vvd
        )

        self.fusion_module_teacher = CrossAttentionFusion(
            hidden_dim=cfg.dim_hidden,
            num_heads=cfg.num_heads,
            norm_inputs=True
        )
        self.fusion_module_student = BiasCrossAttentionFusion(
            hidden_dim=cfg.dim_hidden,
            num_heads=cfg.num_heads,
            bias_proj_dim=cfg.bias_proj_dim
        )

        self.vis_save_dir = "./batch_vis_results"
        os.makedirs(self.vis_save_dir, exist_ok=True)

    def compute_codebook_contrastive_loss(self, dataset_features, teacher_indices, vq_layer):

        bias_module = self.fusion_module_student
        device = dataset_features.device
        B, N, D = dataset_features.shape

        geo_q = bias_module.geo_q_proj(bias_module.norm_q(dataset_features))
        geo_q = F.normalize(geo_q, p=2, dim=-1)

        gt_visual = vq_layer.get_codebook_entry(teacher_indices)

        geo_k_pos = bias_module.geo_k_proj(bias_module.norm_kv(gt_visual))
        geo_k_pos = F.normalize(geo_k_pos, p=2, dim=-1)

        num_negatives = 1024
        codebook_size = self.codebook_size

        neg_indices = torch.randint(0, codebook_size, (num_negatives,), device=device)
        neg_visual = vq_layer.get_codebook_entry(neg_indices)

        geo_k_neg = bias_module.geo_k_proj(bias_module.norm_kv(neg_visual))
        geo_k_neg = F.normalize(geo_k_neg, p=2, dim=-1)

        sim_pos = torch.einsum('bnd,bmd->bnm', geo_q, geo_k_pos)

        best_pos_sim, _ = torch.max(sim_pos, dim=-1, keepdim=True)  # [B, N, 1]

        sim_neg = torch.einsum('bnd,kd->bnk', geo_q, geo_k_neg)

        logits = torch.cat([best_pos_sim, sim_neg], dim=-1)  # [B, N, 1 + K_neg]
        temperature = 0.07
        logits = logits / temperature

        labels = torch.zeros((B, N), dtype=torch.long, device=device)

        loss = F.cross_entropy(logits.view(-1, 1 + num_negatives), labels.view(-1))

        return loss

    def forward(self, points, gt_image=None, inference=False):
        aux_losses = {}

        # A. Encode Dataset Points
        points_emb_ = self.fc_points_(points)
        dataset_features = self.points_encoder(self.fc_points(points))
        # if 1>0:
        #     return dataset_features,dataset_features,aux_losses
        if not inference and gt_image is not None:
            # --- Teacher Path ---
            visual_raw = self.visual_encoder(gt_image)
            quantized_visual_gt, vq_loss_total, gt_indices, perplexity = self.vq_layer(visual_raw)
            aux_losses['vq_loss'] = vq_loss_total
            aux_losses['perplexity'] = perplexity

            # --- Student Path ---
            pred_logits = self.token_predictor(self.points_encoder_(points_emb_))
            token_pred_loss = F.cross_entropy(pred_logits.view(-1, self.codebook_size), gt_indices.view(-1))
            aux_losses['token_loss'] = token_pred_loss

            _, topk_indices = torch.topk(pred_logits, k=self.top_k, dim=-1)
            visual_features_student = self.vq_layer.get_codebook_entry(topk_indices).flatten(1, 2)
            # Alignment Loss
            loss_cl = self.compute_codebook_contrastive_loss(
                dataset_features=dataset_features,
                teacher_indices=gt_indices,
                vq_layer=self.vq_layer  
            )
            aux_losses['contrastive_loss'] = loss_cl

            fused_out_student = self.fusion_module_student(
                dataset_features, visual_features_student
            )

            with torch.no_grad(): 
                _, _, attn_weights_std = self.fusion_module_teacher(
                    dataset_features, visual_features_student
                )

            visual_features_teacher = quantized_visual_gt
            fused_out_teacher, _, _ = self.fusion_module_teacher(dataset_features, visual_features_teacher)

        else:
            pred_logits = self.token_predictor(points_emb_)
            _, topk_indices = torch.topk(pred_logits, k=self.top_k, dim=-1)
            visual_features_student = self.vq_layer.get_codebook_entry(topk_indices).flatten(1, 2)

            aux_losses['vq_loss'] = 0.0
            aux_losses['token_loss'] = 0.0
            aux_losses['perplexity'] = 0.0
            aux_losses['contrastive_loss'] = 0.0

            fused_out_student= self.fusion_module_student(
                dataset_features, visual_features_student
            )
            fused_out_teacher = None

        return fused_out_student, fused_out_teacher, aux_losses

    def predict(self, points):
        fused_out_student, _, _ = self.forward(points, gt_image=None, inference=True)
        return fused_out_student

