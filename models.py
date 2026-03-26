import torch
from torch import nn
import torch.nn.functional as F

import config as CFG
from modules import (
    ImageEncoder,
    ImageEncoder_CLIP,
    ImageEncoder_UNI,
    ImageEncoder_ViT,
    ImageEncoder_ViT_L,
    ImageEncoder_resnet101,
    ImageEncoder_resnet152,
    ProjectionHead,
)


class ResidualMLPBlock(nn.Module):
    """带 LayerNorm 的残差 MLP Block，用于表达域特征建模。"""

    def __init__(self, dim: int, dropout: float = CFG.dropout):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return residual + x


class SpotEncoder(nn.Module):
    """
    显式表达域编码器：
    - 输入 3467 维 reduced_expression
    - 经过多层线性映射 + GELU + LayerNorm + 残差建模共表达关系
    - 输出仍为 3467 维，再交给 ProjectionHead
    """

    def __init__(
        self,
        input_dim: int = CFG.spot_embedding,
        hidden_dim: int = 2048,
        num_res_blocks: int = 2,
        dropout: float = CFG.dropout,
    ):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        self.blocks = nn.ModuleList(
            [ResidualMLPBlock(hidden_dim, dropout=dropout) for _ in range(num_res_blocks)]
        )

        self.output_proj = nn.Linear(hidden_dim, input_dim)
        self.output_dropout = nn.Dropout(dropout)
        self.output_norm = nn.LayerNorm(input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.input_norm(x)
        x = self.input_proj(x)

        for block in self.blocks:
            x = block(x)

        x = self.output_proj(x)
        x = self.output_dropout(x)

        # 与原始表达向量做残差融合，保留一阶表达强度信息。
        x = x + residual
        x = self.output_norm(x)
        return x


class BaseCLIPModel(nn.Module):
    """统一的双模态对比学习骨架，不同 backbone 仅替换 image_encoder。"""

    def __init__(
        self,
        image_encoder: nn.Module,
        image_embedding: int,
        temperature: float = CFG.temperature,
        spot_embedding: int = CFG.spot_embedding,
        use_spot_encoder: bool = CFG.use_spot_encoder,
    ):
        super().__init__()
        self.image_encoder = image_encoder
        # 默认关闭 SpotEncoder，使表达分支更接近原始 BLEEP baseline:
        # reduced_expression -> spot_projection。
        # 同时保留 SpotEncoder 作为可选增强塔用于后续消融实验。
        self.use_spot_encoder = bool(use_spot_encoder)
        self.spot_encoder = SpotEncoder(input_dim=spot_embedding) if self.use_spot_encoder else None

        self.image_projection = ProjectionHead(embedding_dim=image_embedding)
        self.spot_projection = ProjectionHead(embedding_dim=spot_embedding)

        self.temperature = temperature

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        return self.image_encoder(image)

    def encode_spot(self, reduced_expression: torch.Tensor) -> torch.Tensor:
        if self.use_spot_encoder:
            if self.spot_encoder is None:
                raise RuntimeError("`use_spot_encoder=True` but `spot_encoder` is not initialized.")
            return self.spot_encoder(reduced_expression)
        return reduced_expression

    def compute_loss(
        self,
        image_embeddings: torch.Tensor,
        spot_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        logits = (spot_embeddings @ image_embeddings.T) / self.temperature
        images_similarity = image_embeddings @ image_embeddings.T
        spots_similarity = spot_embeddings @ spot_embeddings.T

        targets = F.softmax(
            ((images_similarity + spots_similarity) / 2.0) / self.temperature,
            dim=-1,
        )

        spots_loss = cross_entropy(logits, targets, reduction="none")
        images_loss = cross_entropy(logits.T, targets.T, reduction="none")
        loss = (images_loss + spots_loss) / 2.0
        return loss.mean()

    def forward(self, batch, return_features: bool = False):
        image_features = self.encode_image(batch["image"])
        spot_features = self.encode_spot(batch["reduced_expression"])

        image_embeddings = self.image_projection(image_features)
        spot_embeddings = self.spot_projection(spot_features)

        loss = self.compute_loss(image_embeddings=image_embeddings, spot_embeddings=spot_embeddings)

        if return_features:
            return {
                "loss": loss,
                "image_features": image_features,
                "spot_features": spot_features,
                "image_embeddings": image_embeddings,
                "spot_embeddings": spot_embeddings,
            }
        return loss


class CLIPModel(BaseCLIPModel):
    def __init__(
        self,
        temperature: float = CFG.temperature,
        image_embedding: int = CFG.image_embedding,
        spot_embedding: int = CFG.spot_embedding,
        use_spot_encoder: bool = CFG.use_spot_encoder,
    ):
        super().__init__(
            image_encoder=ImageEncoder(),
            image_embedding=image_embedding,
            temperature=temperature,
            spot_embedding=spot_embedding,
            use_spot_encoder=use_spot_encoder,
        )


class CLIPModel_ViT(BaseCLIPModel):
    def __init__(
        self,
        temperature: float = CFG.temperature,
        image_embedding: int = 768,
        spot_embedding: int = CFG.spot_embedding,
        use_spot_encoder: bool = CFG.use_spot_encoder,
    ):
        super().__init__(
            image_encoder=ImageEncoder_ViT(),
            image_embedding=image_embedding,
            temperature=temperature,
            spot_embedding=spot_embedding,
            use_spot_encoder=use_spot_encoder,
        )


class CLIPModel_CLIP(BaseCLIPModel):
    def __init__(
        self,
        temperature: float = CFG.temperature,
        image_embedding: int = 768,
        spot_embedding: int = CFG.spot_embedding,
        use_spot_encoder: bool = CFG.use_spot_encoder,
    ):
        super().__init__(
            image_encoder=ImageEncoder_CLIP(),
            image_embedding=image_embedding,
            temperature=temperature,
            spot_embedding=spot_embedding,
            use_spot_encoder=use_spot_encoder,
        )


class CLIPModel_ViT_L(BaseCLIPModel):
    def __init__(
        self,
        temperature: float = CFG.temperature,
        image_embedding: int = 1024,
        spot_embedding: int = CFG.spot_embedding,
        use_spot_encoder: bool = CFG.use_spot_encoder,
    ):
        super().__init__(
            image_encoder=ImageEncoder_ViT_L(),
            image_embedding=image_embedding,
            temperature=temperature,
            spot_embedding=spot_embedding,
            use_spot_encoder=use_spot_encoder,
        )


class CLIPModel_resnet101(BaseCLIPModel):
    def __init__(
        self,
        temperature: float = CFG.temperature,
        image_embedding: int = 2048,
        spot_embedding: int = CFG.spot_embedding,
        use_spot_encoder: bool = CFG.use_spot_encoder,
    ):
        super().__init__(
            image_encoder=ImageEncoder_resnet101(),
            image_embedding=image_embedding,
            temperature=temperature,
            spot_embedding=spot_embedding,
            use_spot_encoder=use_spot_encoder,
        )


class CLIPModel_resnet152(BaseCLIPModel):
    def __init__(
        self,
        temperature: float = CFG.temperature,
        image_embedding: int = 2048,
        spot_embedding: int = CFG.spot_embedding,
        use_spot_encoder: bool = CFG.use_spot_encoder,
    ):
        super().__init__(
            image_encoder=ImageEncoder_resnet152(),
            image_embedding=image_embedding,
            temperature=temperature,
            spot_embedding=spot_embedding,
            use_spot_encoder=use_spot_encoder,
        )


class CLIPModel_UNI(BaseCLIPModel):
    def __init__(
        self,
        uni_model_dir: str,
        temperature: float = CFG.temperature,
        image_embedding: int = 1024,
        spot_embedding: int = CFG.spot_embedding,
        use_spot_encoder: bool = CFG.use_spot_encoder,
    ):
        super().__init__(
            image_encoder=ImageEncoder_UNI(model_dir=uni_model_dir),
            image_embedding=image_embedding,
            temperature=temperature,
            spot_embedding=spot_embedding,
            use_spot_encoder=use_spot_encoder,
        )


def cross_entropy(preds, targets, reduction="none"):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    if reduction == "mean":
        return loss.mean()
    raise ValueError(f"Unsupported reduction: {reduction}")


if __name__ == "__main__":
    images = torch.randn(8, 3, 224, 224)
    reduced_expression = torch.randn(8, CFG.spot_embedding)
    batch = {
        "image": images,
        "reduced_expression": reduced_expression,
    }

    clip_model = CLIPModel()
    loss = clip_model(batch)
    print(f"loss={loss.item():.6f}")
