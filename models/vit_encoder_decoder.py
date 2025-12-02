import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from models.vitbackbone import ViTEncoder

class TransformerDecoderLayerWithAttn(nn.TransformerDecoderLayer):
    """
    自定义 DecoderLayer，用于在 forward 过程中保存 attention weights。
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.self_attn_map = None
        self.cross_attn_map = None
        # Norms of the updates (what is added to the stream)
        self.sa_update_norm = None
        self.ca_update_norm = None
        self.ffn_update_norm = None

    def _sa_block(self, x, attn_mask, key_padding_mask, is_causal=False):
        # 强制 need_weights=True
        x, weights = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=True,
                           is_causal=is_causal)
        self.self_attn_map = weights
        output = self.dropout1(x)
        self.sa_update_norm = output.norm(p=2, dim=-1).detach()
        return output

    def _mha_block(self, x, mem, attn_mask, key_padding_mask, is_causal=False):
        # 强制 need_weights=True
        x, weights = self.multihead_attn(x, mem, mem,
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask,
                                need_weights=True,
                                is_causal=is_causal)
        self.cross_attn_map = weights
        output = self.dropout2(x)
        self.ca_update_norm = output.norm(p=2, dim=-1).detach()
        return output

    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        output = self.dropout3(x)
        self.ffn_update_norm = output.norm(p=2, dim=-1).detach()
        return output

class CaptionTransformerDecoder(nn.Module):
    """
    标准 Transformer Decoder：
    - token embedding + learned pos embedding
    - nn.TransformerDecoder (batch_first=True)
    - 输出 vocab 概率分布
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        num_layers: int = 4,
        dim_ff: int = 2048,
        max_len: int = 80,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.d_model = d_model
        self.pad_idx = pad_idx
        self.max_len = max_len

        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_emb = nn.Embedding(max_len, d_model)

        # 使用自定义 Layer 以支持 attention 可视化
        decoder_layer = TransformerDecoderLayerWithAttn(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_ff,
            batch_first=True,  # 允许 (B, L, D)
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers,
        )

        self.fc_out = nn.Linear(d_model, vocab_size)

    @property
    def decoder_layers(self):
        return self.decoder.layers

    @staticmethod
    def _generate_square_subsequent_mask(L: int, device: torch.device) -> torch.Tensor:
        """
        生成 Transformer 的上三角 mask，用于 masked self-attention
        形状: (L, L)
        """
        mask = torch.full((L, L), float("-inf"), device=device)
        mask = torch.triu(mask, diagonal=1)
        return mask

    def forward(self, memory: torch.Tensor, captions_in: torch.Tensor) -> torch.Tensor:
        """
        Args:
            memory:     (B, S, d_model)  来自 ViTEncoder
            captions_in:(B, L)           输入序列 (包含 <START>，不包含最后一个 token)

        Returns:
            logits: (B, L, vocab_size)
        """
        B, L = captions_in.shape
        device = captions_in.device

        positions = torch.arange(0, L, device=device).unsqueeze(0).expand(B, L)
        x = self.token_emb(captions_in) + self.pos_emb(positions)  # (B, L, d_model)

        tgt_mask = self._generate_square_subsequent_mask(L, device)  # (L, L)
        tgt_key_padding_mask = captions_in.eq(self.pad_idx)          # (B, L)

        out = self.decoder(
            tgt=x,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=None,  # 图像 token 都是有效的
        )  # (B, L, d_model)

        logits = self.fc_out(out)  # (B, L, vocab_size)
        return logits

class ImageCaptionModel(nn.Module):
    """
    整体 Encoder-Decoder 模型：
    - Encoder: ViTEncoder
    - Decoder: CaptionTransformerDecoder
    训练时使用 teacher forcing：
        inputs : <START> ... token_{T-1}
        targets: token_1 ... <END>
    """

    def __init__(
        self,
        vocab_size: int,
        pad_idx: int = 0,
        start_idx: int = 1,
        end_idx: int = 2,
        d_model: int = 512,
        n_heads: int = 8,
        num_layers: int = 4,
        max_len: int = 80,
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

    def forward(self, images: torch.Tensor, captions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            images:   (B, 3, 224, 224)
            captions: (B, L) 已含 <START> ... <END> 并 pad

        Returns:
            logits:   (B, L-1, vocab_size)
            targets:  (B, L-1)
        """
        # 1) ViT Encoder
        memory = self.encoder(images)  # (B, S, d_model)

        # 2) 准备 decoder 输入/输出
        captions_in = captions[:, :-1]   # 去掉最后一个 token
        captions_tgt = captions[:, 1:]   # 去掉第一个 token (<START>)

        # 3) Transformer Decoder
        logits = self.decoder(memory, captions_in)  # (B, L-1, vocab_size)

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


