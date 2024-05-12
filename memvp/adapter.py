import torch
from torch import nn
from torch.nn import functional as F
import memvp
from typing import Optional, Tuple
from torch.cuda.amp import autocast
from clip.model import ResidualAttentionBlock


class Adapter(nn.Module):
    def __init__(
            self,
            in_features=768,
            hidden_dim=8,
    ):
        super().__init__()
        if hidden_dim > 0:
            self.fc1 = nn.Linear(in_features, hidden_dim, bias=False)
            self.fc2 = nn.Linear(hidden_dim, in_features, bias=False)
            self.hidden_dim = hidden_dim
            nn.init.zeros_(self.fc2.weight)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, vis_weight):
        with autocast():
            if vis_weight is not None:
                x = x @ (vis_weight[0] + vis_weight[1]).permute(0, 2, 1)
                x = self.dropout(F.silu(x))
                x = x @ (vis_weight[0] + vis_weight[2])
            else:
                x = self.fc1(x)
                x = self.dropout(F.gelu(x))
                x = self.fc2(x)
        return x


def forward_llama(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor],
                  vis_weight):
    if self.training and self.gradient_checkpointing:
        h = x + torch.utils.checkpoint.checkpoint(self.attention, self.attention_norm(x), start_pos, freqs_cis, mask)
        h_norm = self.ffn_norm(h)
        out = h + torch.utils.checkpoint.checkpoint(self.feed_forward, h_norm) + self.adapter_mlp(h_norm,
                                                                                                  vis_weight) * self.s
    else:
        h = x + self.attention(self.attention_norm(x), start_pos, freqs_cis, mask)
        out = h + self.drop_path(
            self.feed_forward((self.ffn_norm(h))) + self.adapter_mlp(self.ffn_norm(h), vis_weight) * self.s)
    return out


def forward_clip(self, x: torch.Tensor):
    x = x + self.attention(self.ln_1(x))
    x = x + self.mlp(self.ln_2(x)) + self.adapter_mlp(self.ln_2(x), None) * self.s
    return x


def set_Llama_Adapter(model, s=1, gradient_checkpointing=False):
    for _ in model.children():
        if type(_) == memvp.model.TransformerBlock:
            _.adapter_mlp = Adapter(_.dim, hidden_dim=0)
            _.s = s
            _.gradient_checkpointing = gradient_checkpointing
            bound_method = forward_llama.__get__(_, _.__class__)
            setattr(_, 'forward', bound_method)
        elif len(list(_.children())) != 0:
            set_Llama_Adapter(_, s, gradient_checkpointing=gradient_checkpointing)


def set_Clip_Adapter(model, dim=8, s=0.1):
    for _ in model.children():
        if type(_) == ResidualAttentionBlock:
            _.adapter_mlp = Adapter(1024, hidden_dim=dim)
            _.s = s
            bound_method = forward_clip.__get__(_, _.__class__)
            setattr(_, 'forward', bound_method)
        elif len(list(_.children())) != 0:
            set_Clip_Adapter(_, dim, s)
