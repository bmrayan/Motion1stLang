from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class VQVAEConfig:
    input_dim: int
    hidden_dims: List[int]
    embedding_dim: int
    codebook_size: int
    commitment_beta: float = 0.25
    decay: float = 0.99
    eps: float = 1e-5
    dead_code_threshold: float = 2.0


class VectorQuantizerEMA(nn.Module):
    def __init__(
        self,
        codebook_size: int,
        embedding_dim: int,
        decay: float = 0.99,
        eps: float = 1e-5,
        dead_code_threshold: float = 2.0,
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.decay = decay
        self.eps = eps
        self.dead_code_threshold = dead_code_threshold

        embed = torch.randn(codebook_size, embedding_dim)
        self.register_buffer("embedding", embed)
        self.register_buffer("cluster_size", torch.zeros(codebook_size))
        self.register_buffer("embed_avg", embed.clone())
        self.register_buffer("initialized", torch.tensor(False))

    @torch.no_grad()
    def _initialize_from_batch(self, z_flat: torch.Tensor) -> None:
        if bool(self.initialized.item()) or z_flat.numel() == 0:
            return
        if z_flat.shape[0] >= self.codebook_size:
            indices = torch.randperm(z_flat.shape[0], device=z_flat.device)[: self.codebook_size]
            initial = z_flat[indices]
        else:
            repeat = (self.codebook_size + z_flat.shape[0] - 1) // z_flat.shape[0]
            tiled = z_flat.repeat(repeat, 1)
            indices = torch.randperm(tiled.shape[0], device=z_flat.device)[: self.codebook_size]
            initial = tiled[indices]
        self.embedding.copy_(initial)
        self.embed_avg.copy_(initial)
        self.cluster_size.fill_(1.0)
        self.initialized.fill_(True)

    @torch.no_grad()
    def _reset_unused_codes(self, z_flat: torch.Tensor, used_counts: torch.Tensor, threshold: float = 1.0) -> None:
        if z_flat.numel() == 0:
            return
        unused = used_counts < threshold
        num_unused = int(unused.sum().item())
        if num_unused == 0:
            return
        if z_flat.shape[0] >= num_unused:
            replace_idx = torch.randperm(z_flat.shape[0], device=z_flat.device)[:num_unused]
        else:
            replace_idx = torch.randint(0, z_flat.shape[0], (num_unused,), device=z_flat.device)
        replacements = z_flat[replace_idx]
        self.embedding[unused] = replacements
        self.embed_avg[unused] = replacements
        self.cluster_size[unused] = 1.0

    @torch.no_grad()
    def initialize_codebook(self, embeddings: torch.Tensor) -> None:
        embeddings = embeddings.detach()
        if embeddings.shape[0] < self.codebook_size:
            repeat = (self.codebook_size + embeddings.shape[0] - 1) // embeddings.shape[0]
            embeddings = embeddings.repeat(repeat, 1)
        embeddings = embeddings[: self.codebook_size]
        self.embedding.copy_(embeddings)
        self.embed_avg.copy_(embeddings)
        self.cluster_size.fill_(1.0)
        self.initialized.fill_(True)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_flat = z.reshape(-1, self.embedding_dim)
        self._initialize_from_batch(z_flat)
        distances = (
            z_flat.pow(2).sum(dim=1, keepdim=True)
            - 2.0 * z_flat @ self.embedding.t()
            + self.embedding.pow(2).sum(dim=1, keepdim=True).t()
        )
        indices = distances.argmin(dim=1)
        quantized = F.embedding(indices, self.embedding).view_as(z)

        if self.training:
            used_counts = torch.bincount(indices, minlength=self.codebook_size).to(z_flat.dtype)
            embed_sum = torch.zeros_like(self.embed_avg)
            embed_sum.index_add_(0, indices, z_flat.detach())
            self.cluster_size.mul_(self.decay).add_(used_counts, alpha=1.0 - self.decay)
            self.embed_avg.mul_(self.decay).add_(embed_sum, alpha=1.0 - self.decay)

            n = self.cluster_size.sum()
            smoothed = (self.cluster_size + self.eps) / (n + self.codebook_size * self.eps) * n
            self.embedding.copy_(self.embed_avg / smoothed.unsqueeze(1))
            self._reset_unused_codes(
                z_flat.detach(),
                self.cluster_size.detach(),
                threshold=self.dead_code_threshold,
            )

        quantized = z + (quantized - z).detach()
        return quantized, indices.view(z.shape[:-1]), torch.empty(0, device=z.device)


class MLPVQVAE(nn.Module):
    def __init__(self, config: VQVAEConfig):
        super().__init__()
        self.config = config
        encoder_layers: List[nn.Module] = []
        prev = config.input_dim
        for hidden in config.hidden_dims:
            encoder_layers.append(nn.Linear(prev, hidden))
            encoder_layers.append(nn.ReLU())
            prev = hidden
        encoder_layers.append(nn.Linear(prev, config.embedding_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers: List[nn.Module] = []
        prev = config.embedding_dim
        mirrored = list(reversed(config.hidden_dims))
        for hidden in mirrored:
            decoder_layers.append(nn.Linear(prev, hidden))
            decoder_layers.append(nn.ReLU())
            prev = hidden
        decoder_layers.append(nn.Linear(prev, config.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

        self.quantizer = VectorQuantizerEMA(
            codebook_size=config.codebook_size,
            embedding_dim=config.embedding_dim,
            decay=config.decay,
            eps=config.eps,
            dead_code_threshold=config.dead_code_threshold,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_e = self.encoder(x)
        z_q, indices, encodings = self.quantizer(z_e)
        recon = self.decoder(z_q)
        commitment_loss = self.config.commitment_beta * F.mse_loss(z_e, z_q.detach())
        recon_loss = F.mse_loss(recon, x)
        loss = recon_loss + commitment_loss
        return loss, recon_loss.detach(), indices, recon

    @torch.no_grad()
    def encode_indices(self, x: torch.Tensor) -> torch.Tensor:
        z_e = self.encoder(x)
        _, indices, _ = self.quantizer(z_e)
        return indices

    @torch.no_grad()
    def encode_latents(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    @torch.no_grad()
    def initialize_codebook(self, embeddings: torch.Tensor) -> None:
        self.quantizer.initialize_codebook(embeddings)
