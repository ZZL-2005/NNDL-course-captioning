# vit_encoder_decoder_prenorm.py
import torch
import torch.nn as nn
from models.vitbackbone import ViTEncoder
class PreNormTransformerDecoderLayer(nn.Module):
    """
    Pre-Norm Transformer Decoder Layer with full residual stream recording.

    Structure:
        x0 = x
        x1 = x0 + ΔSA
        x2 = x1 + ΔCA
        x3 = x2 + ΔFFN

    When record=True, only the LAST TOKEN is recorded.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dim_ff: int,
        dropout: float = 0.0,
        record: bool = False,
    ):
        super().__init__()

        self.record = record

        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.ReLU(),
            nn.Linear(dim_ff, d_model),
            nn.Dropout(dropout),
        )

        self.norm_sa = nn.LayerNorm(d_model)
        self.norm_ca = nn.LayerNorm(d_model)
        self.norm_ff = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

        self.clear_records()

    def clear_records(self):
        # residual stream (last token)
        self.x_init = None      # before SA
        self.x_after_sa = None  # after SA
        self.x_after_ca = None  # after CA
        self.x_final = None     # after FFN

        # updates (Δx)
        self.last_sa = None
        self.last_ca = None
        self.last_ffn = None

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        tgt_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:

        # ===== Initial residual =====
        if self.record:
            self.x_init = x[:, -1].detach()

        # ===== Self-Attention =====
        sa_input = self.norm_sa(x)
        sa_out, _ = self.self_attn(
            sa_input, sa_input, sa_input,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
            need_weights=False,
        )

        if self.record:
            self.last_sa = sa_out[:, -1].detach()

        x = x + self.dropout(sa_out)

        if self.record:
            self.x_after_sa = x[:, -1].detach()

        # ===== Cross-Attention =====
        ca_input = self.norm_ca(x)
        ca_out, _ = self.cross_attn(
            ca_input, memory, memory,
            need_weights=False,
        )

        if self.record:
            self.last_ca = ca_out[:, -1].detach()

        x = x + self.dropout(ca_out)

        if self.record:
            self.x_after_ca = x[:, -1].detach()

        # ===== Feed-Forward Network =====
        ffn_input = self.norm_ff(x)
        ffn_out = self.ffn(ffn_input)

        if self.record:
            self.last_ffn = ffn_out[:, -1].detach()

        x = x + ffn_out

        if self.record:
            self.x_final = x[:, -1].detach()

        return x


class CaptionTransformerDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        num_layers: int = 4,
        dim_ff: int = 2048,
        max_len: int = 130,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.max_len = max_len

        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_emb = nn.Embedding(max_len, d_model)

        self.layers = nn.ModuleList([
            PreNormTransformerDecoderLayer(
                d_model=d_model,
                n_heads=n_heads,
                dim_ff=dim_ff,
            )
            for _ in range(num_layers)
        ])

        self.fc_out = nn.Linear(d_model, vocab_size)

    @staticmethod
    def generate_square_subsequent_mask(L: int, device):
        mask = torch.triu(
            torch.full((L, L), float("-inf"), device=device),
            diagonal=1
        )
        return mask

    def forward(self, memory: torch.Tensor, captions_in: torch.Tensor):
        """
        Args:
            memory:      (B, S, d_model)
            captions_in: (B, L)
        Returns:
            logits:      (B, L, vocab_size)
        """
        B, L = captions_in.shape
        device = captions_in.device

        positions = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
        x = self.token_emb(captions_in) + self.pos_emb(positions)

        tgt_mask = self.generate_square_subsequent_mask(L, device)
        tgt_key_padding_mask = captions_in.eq(self.pad_idx)

        for layer in self.layers:
            x = layer(
                x,
                memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
            )

        return self.fc_out(x)

class ImageCaptionModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        pad_idx: int = 0,
        start_idx: int = 1,
        end_idx: int = 2,
        d_model: int = 512,
        n_heads: int = 8,
        num_layers: int = 4,
        max_len: int = 130,
        num_img_tokens: int | None = None,
        freeze_encoder: bool = False,
    ):
        super().__init__()

        self.encoder = ViTEncoder(
            d_model=d_model,
            num_img_tokens=num_img_tokens,
            freeze=freeze_encoder,
        )

        self.decoder = CaptionTransformerDecoder(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            num_layers=num_layers,
            max_len=max_len,
            pad_idx=pad_idx,
        )

        self.pad_idx = pad_idx
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.max_len = max_len

    def forward(self, images, captions):
        """
        captions: (B, L)  <START> ... <END> PAD
        """
        memory = self.encoder(images)

        captions_in = captions[:, :-1]
        captions_tgt = captions[:, 1:]

        logits = self.decoder(memory, captions_in)

        return logits, captions_tgt
    @torch.no_grad()
    def greedy_decode(self, images: torch.Tensor) -> torch.Tensor:
        """
        推理用：贪心解码
        Args:
            images: (B, 3, 224, 224)

        Returns:
            ys: (B, max_len) 生成的序列 (包含 <START> 和 <END>，中间可能有 PAD)
        """
        self.eval()
        device = images.device
        B = images.size(0)

        memory = self.encoder(images)  # (B, S, d_model)

        # 初始化生成序列：全是 <START>
        ys = torch.full((B, 1), self.start_idx, dtype=torch.long, device=device)

        for _ in range(self.max_len - 1):
            logits = self.decoder(memory, ys)   # (B, t, vocab)
            next_logits = logits[:, -1, :]      # (B, vocab)
            next_tokens = next_logits.argmax(dim=-1)  # (B,)

            ys = torch.cat([ys, next_tokens.unsqueeze(1)], dim=1)  # (B, t+1)

            # 如果全 batch 都生成了 <END>，可以提前结束（不是必须）
            if (next_tokens == self.end_idx).all():
                break

        return ys
