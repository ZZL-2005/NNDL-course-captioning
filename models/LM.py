import torch
import torch.nn as nn


class OneHotLanguageModel(nn.Module):
    """
    直接使用 one-hot 作为 token embedding，不使用 nn.Embedding
    vocab_size = d_model = one-hot 维度，例如 109
    """

    def __init__(
        self,
        vocab_size: int,
        max_len: int = 80,
        n_heads: int = 8,
        num_layers: int = 4,
        dim_ff: int = 2048,
        pad_idx: int = 0
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = vocab_size        # embedding dim = vocab_size
        self.max_len = max_len
        self.pad_idx = pad_idx

        # 位置编码维度也必须 = d_model = vocab_size
        self.pos_emb = nn.Embedding(max_len, vocab_size)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=vocab_size,
            nhead=n_heads,
            dim_feedforward=dim_ff,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # output projection
        self.output_proj = nn.Linear(vocab_size, vocab_size)

        self._init_params()

    def _init_params(self):
        nn.init.normal_(self.pos_emb.weight, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def _build_causal_mask(self, L: int, device):
        mask = torch.full((L, L), float("-inf"), device=device)
        return torch.triu(mask, diagonal=1)

    def _build_padding_mask(self, caps: torch.Tensor):
        return (caps == self.pad_idx)

    def forward(self, caps):
        """
        caps: (B, L)  token ids
        返回:
            logits: (B, L, vocab_size)
        """

        B, L = caps.shape
        device = caps.device

        # (B, L, vocab_size) one-hot 编码
        x = torch.nn.functional.one_hot(caps, num_classes=self.vocab_size).float()

        # 位置编码
        pos_ids = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
        x = x + self.pos_emb(pos_ids)

        # causal mask & pad mask
        tgt_mask = self._build_causal_mask(L, device)
        tgt_key_padding_mask = self._build_padding_mask(caps)

        # dummy memory
        memory = torch.zeros(B, 1, self.d_model, device=device)
        mem_pad = torch.zeros(B, 1, dtype=torch.bool, device=device)

        h = self.decoder(
            tgt=x,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=mem_pad
        )  # (B, L, vocab)

        logits = self.output_proj(h)  # (B, L, vocab)
        return logits

class AdjustableEmbeddingLM(nn.Module):
    """
    输入 one-hot → Linear(vocab_size → emb_dim)
    支持 freeze_input: 冻结 embedding，不更新权重
    """

    def __init__(
        self,
        vocab_size: int,
        emb_dim: int = 109,
        max_len: int = 80,
        n_heads: int = 1,
        num_layers: int = 4,
        dim_ff: int = 2048,
        pad_idx: int = 0,
        freeze_input: bool = False
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.max_len = max_len
        self.pad_idx = pad_idx
        self.freeze_input = freeze_input

        # ------- input embedding (可冻结) -------
        self.input_proj = nn.Linear(vocab_size, emb_dim)
        if freeze_input:
            for p in self.input_proj.parameters():
                p.requires_grad = False

        # ------- 位置编码 -------
        self.pos_emb = nn.Embedding(max_len, emb_dim)

        # ------- decoder -------
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=emb_dim,
            nhead=n_heads,
            dim_feedforward=dim_ff,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # ------- 输出投影 -------
        self.output_proj = nn.Linear(emb_dim, vocab_size)

        self._init_params()

    def _init_params(self):
        nn.init.normal_(self.pos_emb.weight, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def _build_causal_mask(self, L, device):
        mask = torch.full((L, L), float("-inf"), device=device)
        return torch.triu(mask, diagonal=1)

    def _build_padding_mask(self, caps):
        return caps == self.pad_idx

    def forward(self, caps):
        B, L = caps.shape
        device = caps.device

        # ------- one-hot -------
        x = torch.nn.functional.one_hot(caps, num_classes=self.vocab_size).float()

        # ------- embedding (可冻结) -------
        x = self.input_proj(x)  # 若 freeze_input=True，此层不更新参数

        # ------- pos embedding -------
        pos_ids = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
        x = x + self.pos_emb(pos_ids)

        tgt_mask = self._build_causal_mask(L, device)
        tgt_pad = self._build_padding_mask(caps)
        memory = torch.zeros(B, 1, self.emb_dim, device=device)
        mem_pad = torch.zeros(B, 1, dtype=torch.bool, device=device)

        h = self.decoder(
            tgt=x,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_pad,
            memory_key_padding_mask=mem_pad
        )

        return self.output_proj(h)
