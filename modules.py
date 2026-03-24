import os
import colorsys
from typing import Optional
import torch
from torch import nn
import numpy as np

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None
import config as CFG

# `timm` is required for creating image encoders.
# In some environments (e.g. notebook kernel), the active Python may not have `timm`
# installed even though it's available under this repo's local venv.
try:
    import timm  # type: ignore
except ModuleNotFoundError as e:
    import sys
    import glob

    # Best-effort: add this repo's local venv site-packages to sys.path.
    local_sitepkgs = glob.glob(
        os.path.join(os.path.dirname(__file__), ".venv", "lib", "python*", "site-packages")
    )
    if local_sitepkgs:
        # Put it at the end so it doesn't override the active environment's
        # torch/torchvision/PIL packages (which may have compiled extensions).
        sys.path.append(local_sitepkgs[0])

    try:
        import timm  # type: ignore
    except ModuleNotFoundError:
        raise e

try:
    from safetensors.torch import load_file as _load_safetensors
except Exception:  # pragma: no cover
    _load_safetensors = None


def _maybe_load_local_safetensors(model: nn.Module, local_path: str) -> bool:
    """
    If `local_path` exists and is a valid safetensors checkpoint, load it into `model`.
    Returns True if loaded, else False.
    """
    if not local_path or not os.path.exists(local_path) or _load_safetensors is None:
        return False

    state_dict = _load_safetensors(local_path)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    # Avoid spamming both ranks: rely on trainer stdout; this is still helpful in single-rank.
    print(
        f"Loaded local weights from {local_path} "
        f"(missing={len(missing)}, unexpected={len(unexpected)}).",
        flush=True,
    )
    return True


class ImageEncoder(nn.Module):
    """
    Encode images to a fixed size vector
    """

    def __init__(
        self, model_name=CFG.model_name, pretrained=CFG.pretrained, trainable=CFG.trainable
    ):
        super().__init__()
        # If local weights exist, do NOT let timm download (often fails on air-gapped machines).
        local_weights = os.environ.get("BLEEP_LOCAL_PRETRAINED")
        if local_weights is None:
            local_weights = os.path.join(os.path.dirname(__file__), "model.safetensors")

        use_local = bool(pretrained) and os.path.exists(local_weights)
        self.model = timm.create_model(
            model_name,
            pretrained=False if use_local else pretrained,
            num_classes=0,
            global_pool="avg",
        )
        if use_local:
            _maybe_load_local_safetensors(self.model, local_weights)
        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        return self.model(x)
    
class ImageEncoder_resnet50(nn.Module):
    """
    Encode images to a fixed size vector
    """

    def __init__(
        self, model_name='resnet50', pretrained=CFG.pretrained, trainable=CFG.trainable
    ):
        super().__init__()
        local_weights = os.environ.get("BLEEP_LOCAL_PRETRAINED")
        if local_weights is None:
            local_weights = os.path.join(os.path.dirname(__file__), "model.safetensors")
        use_local = bool(pretrained) and os.path.exists(local_weights)
        self.model = timm.create_model(
            model_name,
            pretrained=False if use_local else pretrained,
            num_classes=0,
            global_pool="avg",
        )
        if use_local:
            _maybe_load_local_safetensors(self.model, local_weights)
        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        return self.model(x)
    
class ImageEncoder_resnet101(nn.Module):
    """
    Encode images to a fixed size vector
    """

    def __init__(
        self, model_name='resnet101', pretrained=True, trainable=CFG.trainable
    ):
        super().__init__()
        local_weights = os.environ.get("BLEEP_LOCAL_PRETRAINED")
        if local_weights is None:
            local_weights = os.path.join(os.path.dirname(__file__), "model.safetensors")
        use_local = bool(pretrained) and os.path.exists(local_weights)
        self.model = timm.create_model(
            model_name,
            pretrained=False if use_local else pretrained,
            num_classes=0,
            global_pool="avg",
        )
        if use_local:
            _maybe_load_local_safetensors(self.model, local_weights)
        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        return self.model(x)
    
class ImageEncoder_resnet152(nn.Module):
    """
    Encode images to a fixed size vector
    """

    def __init__(
        self, model_name='resnet152', pretrained=True, trainable=CFG.trainable
    ):
        super().__init__()
        local_weights = os.environ.get("BLEEP_LOCAL_PRETRAINED")
        if local_weights is None:
            local_weights = os.path.join(os.path.dirname(__file__), "model.safetensors")
        use_local = bool(pretrained) and os.path.exists(local_weights)
        self.model = timm.create_model(
            model_name,
            pretrained=False if use_local else pretrained,
            num_classes=0,
            global_pool="avg",
        )
        if use_local:
            _maybe_load_local_safetensors(self.model, local_weights)
        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        return self.model(x)
    
class ImageEncoder_ViT(nn.Module):
    """
    Encode images to a fixed size vector
    """

    def __init__(
        self, model_name="vit_base_patch32_224", pretrained=False, trainable=CFG.trainable
    ):
        super().__init__()
        self.model = timm.create_model(
            model_name, pretrained, num_classes=0, global_pool="avg"
        )
        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        return self.model(x)
    

class ImageEncoder_CLIP(nn.Module):
    """
    Encode images to a fixed size vector
    """

    def __init__(
        self, model_name="vit_base_patch32_224_clip_laion2b", pretrained=True, trainable=CFG.trainable
    ):
        super().__init__()
        self.model = timm.create_model(
            model_name, pretrained, num_classes=0, global_pool="avg"
        )
        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        return self.model(x)

class ImageEncoder_ViT_L(nn.Module):
    """
    Encode images to a fixed size vector
    """

    def __init__(
        self, model_name="vit_large_patch32_224_in21k", pretrained=False, trainable=CFG.trainable
    ):
        super().__init__()
        self.model = timm.create_model(
            model_name, pretrained, num_classes=0, global_pool="avg"
        )
        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        return self.model(x)
    

#  'vit_base_patch32_224',
#  'vit_base_patch32_224_clip_laion2b',
#  'vit_base_patch32_224_in21k',
#  'vit_base_patch32_224_sam',
    


# class SpotEncoder(nn.Module):
#     #to change...
#     def __init__(self, model_name=CFG.spot_encoder_model, pretrained=CFG.pretrained, trainable=CFG.trainable):
#         super().__init__()
#         if pretrained:
#             self.model = DistilBertModel.from_pretrained(model_name)
#         else:
#             self.model = DistilBertModel(config=DistilBertConfig())
            
#         for p in self.model.parameters():
#             p.requires_grad = trainable

#         # we are using the CLS token hidden representation as the sentence's embedding
#         self.target_token_idx = 0

#     def forward(self, input_ids, attention_mask):
#         output = self.model(input_ids=input_ids, attention_mask=attention_mask)
#         last_hidden_state = output.last_hidden_state
#         return last_hidden_state[:, self.target_token_idx, :]



class ProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim=CFG.projection_dim,
        dropout=CFG.dropout
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)
    
    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x


class AdaptiveRegionGenerator(nn.Module):
    """
    CARE-style Adaptive Region Generator (ARG) for pure HE modality.

    Inputs:
        F: (N, d) patch features
        T: (N, 2) physical coordinates
    Outputs:
        Z_HE_valid: (V, d) adaptive region features after ARSA
        region_indices: (N,) each patch -> valid region index in [0, V-1]
    """

    def __init__(
        self,
        feature_dim: int,
        grid_size: int = 8,
        topk: int = 3,
        arsa_num_heads: int = 8,
        arsa_dropout: float = 0.1,
    ):
        super().__init__()
        if grid_size <= 0:
            raise ValueError(f"`grid_size` must be positive, got {grid_size}.")
        if topk <= 0:
            raise ValueError(f"`topk` must be positive, got {topk}.")
        if feature_dim % arsa_num_heads != 0:
            raise ValueError(
                f"`feature_dim` ({feature_dim}) must be divisible by `arsa_num_heads` "
                f"({arsa_num_heads})."
            )

        self.feature_dim = feature_dim
        self.grid_size = grid_size
        self.topk = topk

        self.cls_ar_token = nn.Parameter(torch.zeros(1, 1, feature_dim))
        nn.init.trunc_normal_(self.cls_ar_token, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=arsa_num_heads,
            dropout=arsa_dropout,
            batch_first=True,
            activation="gelu",
        )
        self.arsa_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

    def _init_grid_anchors(self, F: torch.Tensor, T: torch.Tensor):
        """
        Step 1:
            Build k x k anchors from coordinate range and initialize region
            center features by averaging assigned patches in each grid cell.
        Returns:
            P: (R, 2) anchor coordinates
            G: (R, d) anchor features
        """
        N, d = F.shape
        k = self.grid_size
        device = F.device
        dtype = T.dtype
        eps = torch.finfo(dtype).eps

        t_min = T.min(dim=0).values
        t_max = T.max(dim=0).values
        span = (t_max - t_min).clamp_min(eps)

        x_edges = torch.linspace(t_min[0], t_min[0] + span[0], steps=k + 1, device=device, dtype=dtype)
        y_edges = torch.linspace(t_min[1], t_min[1] + span[1], steps=k + 1, device=device, dtype=dtype)
        x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
        y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])

        xx, yy = torch.meshgrid(x_centers, y_centers, indexing="ij")
        P = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)  # (R, 2), R = k*k

        x_bin = torch.bucketize(T[:, 0].contiguous(), x_edges[1:-1], right=False)
        y_bin = torch.bucketize(T[:, 1].contiguous(), y_edges[1:-1], right=False)
        init_region_id = x_bin * k + y_bin  # (N,)

        R = k * k
        G = F.new_zeros(R, d)
        counts = F.new_zeros(R)
        G.index_add_(0, init_region_id, F)
        counts.index_add_(0, init_region_id, F.new_ones(N))
        G = G / counts.clamp_min(1.0).unsqueeze(-1)

        return P, G

    def forward(self, F: torch.Tensor, T: torch.Tensor):
        if F.ndim != 2:
            raise ValueError(f"`F` must be 2D (N, d), got shape {tuple(F.shape)}.")
        if T.ndim != 2 or T.size(-1) != 2:
            raise ValueError(f"`T` must be 2D (N, 2), got shape {tuple(T.shape)}.")
        if F.size(0) != T.size(0):
            raise ValueError(
                f"`F` and `T` must share N, got F={tuple(F.shape)}, T={tuple(T.shape)}."
            )
        if F.size(1) != self.feature_dim:
            raise ValueError(
                f"Feature dim mismatch: got F.shape[1]={F.size(1)}, "
                f"expected {self.feature_dim}."
            )

        N, d = F.shape
        if N == 0:
            empty_z = F.new_zeros(0, d)
            empty_idx = torch.empty(0, dtype=torch.long, device=F.device)
            return empty_z, empty_idx

        T = T.to(dtype=F.dtype)

        # Step 1: initialize anchors and anchor features from a k x k grid.
        P, G = self._init_grid_anchors(F, T)  # P: (R, 2), G: (R, d)
        R = P.size(0)

        # Step 2: soft spatial inclusion matrix and Top-K candidates.
        dist = torch.cdist(T, P, p=2)  # (N, R)
        max_dist_per_anchor = dist.max(dim=0, keepdim=True).values.clamp_min(1e-12)  # (1, R)
        mu = 1.0 - dist / max_dist_per_anchor  # (N, R)

        k_eff = min(self.topk, R)
        mu_topk, candidate_idx = torch.topk(mu, k=k_eff, dim=1, largest=True, sorted=True)  # (N, K)

        # Step 3: semantic similarity + final hard assignment.
        candidate_G = G[candidate_idx]  # (N, K, d)
        F_expanded = F.unsqueeze(1).expand(-1, k_eff, -1)  # (N, K, d)
        s = torch.nn.functional.cosine_similarity(F_expanded, candidate_G, dim=-1, eps=1e-8)  # (N, K)
        rho = torch.softmax(s, dim=1)  # (N, K)
        w = rho * mu_topk  # (N, K)

        best_local = w.argmax(dim=1)  # (N,)
        row_ids = torch.arange(N, device=F.device)
        assigned_anchor = candidate_idx[row_ids, best_local]  # (N,) in [0, R-1]

        # Convert raw anchor ids to compact valid-region ids [0, V-1].
        valid_anchor_ids, region_indices = torch.unique(
            assigned_anchor, sorted=True, return_inverse=True
        )
        V = valid_anchor_ids.numel()

        # Step 4 (ARSA): region-wise aggregation with a learnable [CLS]_AR token
        # and a 1-layer Transformer encoder.
        region_counts = torch.bincount(region_indices, minlength=V)
        max_region_len = int(region_counts.max().item())
        seq_len = 1 + max_region_len

        tokens = F.new_zeros(V, seq_len, d)
        tokens[:, 0:1, :] = self.cls_ar_token.expand(V, -1, -1)

        order = torch.argsort(region_indices)
        sorted_region = region_indices[order]
        sorted_F = F[order]

        starts = torch.cumsum(region_counts, dim=0) - region_counts
        repeated_starts = torch.repeat_interleave(starts, region_counts)
        pos_in_region = torch.arange(N, device=F.device) - repeated_starts
        tokens[sorted_region, 1 + pos_in_region, :] = sorted_F

        valid_token_mask = (
            torch.arange(max_region_len, device=F.device).unsqueeze(0)
            < region_counts.unsqueeze(1)
        )  # (V, max_region_len)
        padding_mask = torch.ones(V, seq_len, dtype=torch.bool, device=F.device)
        padding_mask[:, 0] = False
        padding_mask[:, 1:] = ~valid_token_mask

        arsa_out = self.arsa_encoder(tokens, src_key_padding_mask=padding_mask)  # (V, seq_len, d)
        Z_HE_valid = arsa_out[:, 0, :]  # (V, d)

        return Z_HE_valid, region_indices


class DualHypergraphAligner(nn.Module):
    """
    Dual-hypergraph co-alignment module for HE-ST topology-level matching.

    Forward inputs:
        F: (N, d_he) HE patch features
        T: (N, 2) HE patch coords
        region_indices: (N,) ARG region labels in [0, V-1]
        E: (M, d_st) ST spot features
        S: (M, 2) ST spot coords

    Forward output:
        align_loss: scalar tensor (can be used directly by loss.backward()).
    """

    def __init__(
        self,
        he_dim: int,
        st_dim: int,
        d_out: int = 256,
        st_radius: float = 150.0,
        k_hg: int = 5,
        temperature: float = 0.07,
        eps: float = 1e-12,
    ):
        super().__init__()
        if he_dim <= 0 or st_dim <= 0 or d_out <= 0:
            raise ValueError("he_dim/st_dim/d_out must be positive integers.")
        if st_radius <= 0:
            raise ValueError("st_radius must be positive.")
        if k_hg <= 0:
            raise ValueError("k_hg must be positive.")
        if temperature <= 0:
            raise ValueError("temperature must be positive.")

        self.he_dim = he_dim
        self.st_dim = st_dim
        self.d_out = d_out
        self.st_radius = float(st_radius)
        self.k_hg = int(k_hg)
        self.temperature = float(temperature)
        self.eps = float(eps)

        self.he_proj = nn.Linear(he_dim, d_out, bias=False)  # W^p
        self.st_proj = nn.Linear(st_dim, d_out, bias=False)  # W^g
        self.act = nn.GELU()

    def _build_incidence_knn(self, X: torch.Tensor) -> torch.Tensor:
        """
        Build hypergraph incidence matrix H in feature space by KNN.
        Each node defines one hyperedge containing its K nearest neighbors.
        Returns H with shape (n, n).
        """
        n = X.size(0)
        k_eff = min(self.k_hg, n)

        Xn = torch.nn.functional.normalize(X, p=2, dim=1, eps=self.eps)
        sim = Xn @ Xn.t()  # (n, n), cosine similarity matrix
        knn_idx = torch.topk(sim, k=k_eff, dim=1, largest=True, sorted=False).indices  # (n, k_eff)

        H = X.new_zeros(n, n)
        edge_ids = torch.arange(n, device=X.device).unsqueeze(1).expand(n, k_eff)
        H[knn_idx.reshape(-1), edge_ids.reshape(-1)] = 1.0
        return H

    def _hypergraph_conv_pool(self, X: torch.Tensor, proj: nn.Linear) -> torch.Tensor:
        """
        One-layer hypergraph convolution + global average pooling:
        X_new = sigma(Dv^-1/2 H De^-1 H^T Dv^-1/2 X W)
        """
        n = X.size(0)
        if n == 1:
            return self.act(proj(X)).squeeze(0)

        H = self._build_incidence_knn(X)  # (n, n)
        dv = H.sum(dim=1).clamp_min(self.eps)  # (n,)
        de = H.sum(dim=0).clamp_min(self.eps)  # (n,)
        dv_inv_sqrt = torch.rsqrt(dv)  # (n,)
        de_inv = 1.0 / de  # (n,)

        H_de = H * de_inv.unsqueeze(0)
        A = H_de @ H.t()  # H De^-1 H^T
        Theta = dv_inv_sqrt.unsqueeze(1) * A * dv_inv_sqrt.unsqueeze(0)  # Dv^-1/2 A Dv^-1/2

        Xw = proj(X)
        X_new = self.act(Theta @ Xw)
        return X_new.mean(dim=0)

    def forward(
        self,
        F: torch.Tensor,
        T: torch.Tensor,
        region_indices: torch.Tensor,
        E: torch.Tensor,
        S: torch.Tensor,
        return_details: bool = False,
    ):
        if F.ndim != 2:
            raise ValueError(f"F must be (N, d_he), got {tuple(F.shape)}")
        if T.ndim != 2 or T.size(-1) != 2:
            raise ValueError(f"T must be (N, 2), got {tuple(T.shape)}")
        if E.ndim != 2:
            raise ValueError(f"E must be (M, d_st), got {tuple(E.shape)}")
        if S.ndim != 2 or S.size(-1) != 2:
            raise ValueError(f"S must be (M, 2), got {tuple(S.shape)}")
        if region_indices.ndim != 1 or region_indices.size(0) != F.size(0):
            raise ValueError(
                f"region_indices must be (N,), got {tuple(region_indices.shape)} for N={F.size(0)}"
            )
        if F.size(1) != self.he_dim:
            raise ValueError(f"F dim mismatch: got {F.size(1)}, expected {self.he_dim}")
        if E.size(1) != self.st_dim:
            raise ValueError(f"E dim mismatch: got {E.size(1)}, expected {self.st_dim}")

        device = F.device
        dtype = F.dtype
        T = T.to(device=device, dtype=dtype)
        E = E.to(device=device, dtype=dtype)
        S = S.to(device=device, dtype=dtype)
        region_indices = region_indices.to(device=device, dtype=torch.long)

        unique_regions = torch.unique(region_indices, sorted=True)
        z_he_list = []
        z_st_list = []
        valid_region_ids = []

        # Step 1: spatial anchoring and ST recall for each adaptive region.
        for rid in unique_regions.tolist():
            he_mask = region_indices == rid
            if not torch.any(he_mask):
                continue

            Xp = F[he_mask]  # (N_i, d_he)
            Tp = T[he_mask]  # (N_i, 2)
            c_i = Tp.mean(dim=0, keepdim=True)  # (1, 2)

            dist_to_center = torch.cdist(c_i, S, p=2).squeeze(0)  # (M,)
            st_mask = dist_to_center < self.st_radius
            if not torch.any(st_mask):
                continue
            Xg = E[st_mask]  # (M_i, d_st)

            # Step 2 and 3: dual hypergraph encoding.
            z_he_i = self._hypergraph_conv_pool(Xp, self.he_proj)  # (d_out,)
            z_st_i = self._hypergraph_conv_pool(Xg, self.st_proj)  # (d_out,)

            z_he_list.append(z_he_i)
            z_st_list.append(z_st_i)
            valid_region_ids.append(rid)

        # Step 4: symmetric InfoNCE over valid region pairs.
        if len(z_he_list) < 2:
            align_loss = self.he_proj.weight.sum() * 0.0
            if return_details:
                empty = F.new_zeros(0, self.d_out)
                return align_loss, {
                    "z_he": empty,
                    "z_st": empty,
                    "valid_region_ids": torch.tensor(
                        valid_region_ids, device=device, dtype=torch.long
                    ),
                }
            return align_loss

        z_he = torch.stack(z_he_list, dim=0)  # (V_valid, d_out)
        z_st = torch.stack(z_st_list, dim=0)  # (V_valid, d_out)
        z_he = torch.nn.functional.normalize(z_he, p=2, dim=1, eps=self.eps)
        z_st = torch.nn.functional.normalize(z_st, p=2, dim=1, eps=self.eps)

        logits = (z_he @ z_st.t()) / self.temperature
        labels = torch.arange(z_he.size(0), device=device, dtype=torch.long)

        loss_he2st = torch.nn.functional.cross_entropy(logits, labels)
        loss_st2he = torch.nn.functional.cross_entropy(logits.t(), labels)
        align_loss = 0.5 * (loss_he2st + loss_st2he)

        if return_details:
            return align_loss, {
                "z_he": z_he,
                "z_st": z_st,
                "valid_region_ids": torch.tensor(valid_region_ids, device=device, dtype=torch.long),
            }
        return align_loss


def _as_numpy_array(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _to_float01_rgb(image: np.ndarray) -> np.ndarray:
    img = np.asarray(image)
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"`wsi_thumbnail` must have shape (H, W, 3), got {img.shape}.")
    img = img.astype(np.float32, copy=False)
    if img.max() > 1.5:
        img = img / 255.0
    return np.clip(img, 0.0, 1.0)


def visualize_adaptive_regions(
    patch_coords,
    region_indices,
    wsi_thumbnail,
    patch_size,
    sigma: float = 2.0,
    alpha: float = 0.5,
    save_path: Optional[str] = None,
):
    """
    Visualize adaptive regions on thumbnail with:
    1) high-contrast color mask
    2) Gaussian smoothing via scipy.ndimage.gaussian_filter
    3) alpha blending overlay
    """
    try:
        from scipy.ndimage import gaussian_filter
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "visualize_adaptive_regions requires scipy (scipy.ndimage.gaussian_filter)."
        ) from e

    coords = _as_numpy_array(patch_coords)
    rid = _as_numpy_array(region_indices).reshape(-1).astype(np.int64)
    base = _to_float01_rgb(_as_numpy_array(wsi_thumbnail))

    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"`patch_coords` must have shape (N, 2), got {coords.shape}.")
    if coords.shape[0] != rid.shape[0]:
        raise ValueError(
            f"`patch_coords` and `region_indices` must share N, got {coords.shape[0]} vs {rid.shape[0]}."
        )
    if patch_size <= 0:
        raise ValueError(f"`patch_size` must be positive, got {patch_size}.")
    if sigma < 0:
        raise ValueError(f"`sigma` must be non-negative, got {sigma}.")
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"`alpha` must be in [0, 1], got {alpha}.")

    H, W, _ = base.shape
    N = coords.shape[0]
    if N == 0:
        return (base * 255).astype(np.uint8)

    x = coords[:, 0].astype(np.float32)
    y = coords[:, 1].astype(np.float32)
    x_min, x_max = float(x.min()), float(x.max())
    y_min, y_max = float(y.min()), float(y.max())
    x_span = max(x_max - x_min, 1e-6)
    y_span = max(y_max - y_min, 1e-6)
    px = np.clip(((x - x_min) / x_span * (W - 1)).round().astype(np.int32), 0, W - 1)
    py = np.clip(((y - y_min) / y_span * (H - 1)).round().astype(np.int32), 0, H - 1)

    unique_rid = np.unique(rid)
    num_regions = max(len(unique_rid), 1)
    lut = {}
    for i, region_id in enumerate(unique_rid.tolist()):
        hue = (i / num_regions) % 1.0
        sat = 0.9
        val = 0.95
        lut[region_id] = np.array(colorsys.hsv_to_rgb(hue, sat, val), dtype=np.float32)

    color_mask = np.zeros((H, W, 3), dtype=np.float32)
    half = patch_size // 2
    for i in range(N):
        cx, cy = int(px[i]), int(py[i])
        x0, x1 = max(0, cx - half), min(W, cx + half + 1)
        y0, y1 = max(0, cy - half), min(H, cy + half + 1)
        color_mask[y0:y1, x0:x1, :] = lut[int(rid[i])]

    if sigma > 0:
        color_mask = gaussian_filter(color_mask, sigma=(sigma, sigma, 0))

    blended = (1.0 - alpha) * base + alpha * color_mask
    blended = np.clip(blended, 0.0, 1.0)
    blended_u8 = (blended * 255.0).astype(np.uint8)

    if save_path is not None:
        save_visualization_image(blended_u8, save_path)

    return blended_u8


def save_visualization_image(
    image,
    save_path: str,
    dpi: int = 300,
    scale: int = 1,
):
    """
    Save visualization image to PNG/TIFF with optional nearest upscaling.
    """
    if save_path is None:
        raise ValueError("`save_path` must not be None.")
    if scale <= 0:
        raise ValueError(f"`scale` must be positive, got {scale}.")

    img = _as_numpy_array(image)
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"`image` must have shape (H, W, 3), got {img.shape}.")
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    H, W = img.shape[:2]
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    if int(scale) > 1:
        new_size = (W * int(scale), H * int(scale))
        if cv2 is not None:
            img = cv2.resize(img, new_size, interpolation=cv2.INTER_NEAREST)
        else:
            from PIL import Image

            img = np.array(
                Image.fromarray(img, mode="RGB").resize(new_size, resample=Image.NEAREST)
            )

    ext = os.path.splitext(save_path)[1].lower()
    if ext not in (".png", ".tif", ".tiff"):
        raise ValueError(f"Unsupported output extension: {ext}. Use .png/.tif/.tiff.")

    # Prefer PIL to preserve DPI metadata for both PNG and TIFF.
    from PIL import Image

    Image.fromarray(img, mode="RGB").save(save_path, dpi=(dpi, dpi))
