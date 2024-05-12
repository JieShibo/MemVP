from typing import Optional, Tuple
from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F

from torch.nn import Embedding, Linear
import torch
import clip
from torch.cuda.amp import autocast


@dataclass
class ModelArgs:
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    norm_eps: float = 1e-5
    hidden_proj: int = 128
    emb: int = 320
    max_batch_size: int = 32
    max_seq_len: int = 2048
    is_train: bool = True


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
        xq: torch.Tensor,
        xk: torch.Tensor,
        freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.n_local_heads = args.n_heads
        self.head_dim = args.dim // args.n_heads

        # modified bias for reparameterizing
        self.wq = Linear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False
        )
        self.wk = Linear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False
        )
        self.wv = Linear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False
        )
        self.wo = Linear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=False
        )
        if not args.is_train:
            self.cache_k = torch.zeros(
                (args.max_batch_size, args.max_seq_len, self.n_local_heads, self.head_dim)
            ).cuda()
            self.cache_v = torch.zeros(
                (args.max_batch_size, args.max_seq_len, self.n_local_heads, self.head_dim)
            ).cuda()
        else:
            self.cache_k = None
            self.cache_v = None

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        if self.cache_k is not None:
            self.cache_k = self.cache_k.to(xq)
            self.cache_v = self.cache_v.to(xq)

            self.cache_k[:bsz, start_pos: start_pos + seqlen] = xk
            self.cache_v[:bsz, start_pos: start_pos + seqlen] = xv

            keys = self.cache_k[:bsz, : start_pos + seqlen]
            values = self.cache_v[:bsz, : start_pos + seqlen]

        else:
            keys = xk
            values = xv
        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, slen, cache_len + slen)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)  # (bs, n_local_heads, slen, head_dim)
        output = output.transpose(
            1, 2
        ).contiguous().view(bsz, seqlen, -1)

        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(
            self,
            dim: int,
            hidden_dim: int,
            multiple_of: int,
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = Linear(
            dim, hidden_dim, bias=False
        )
        self.w2 = Linear(
            hidden_dim, dim, bias=False
        )
        self.w3 = Linear(
            dim, hidden_dim, bias=False
        )

    def forward(self, x):
        x = self.w2(F.silu(self.w1(x), inplace=False) * self.w3(x))
        return x

class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim, hidden_dim=4 * args.dim, multiple_of=args.multiple_of
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.drop_path = nn.Identity()

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        h = x + self.attention(self.attention_norm(x), start_pos, freqs_cis, mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Projector(nn.Module):
    """ Pytorch Implemention of RepAdapter for 1d tensor"""

    def __init__(
            self,
            in_features=768,
            hidden_dim=128,
            out_features=4096
    ):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_features)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        with autocast():
            x = self.fc2(F.silu(self.fc1(x)))
        return x


class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers
        self.tok_embeddings = Embedding(
            params.vocab_size, params.dim
        )

        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=0)

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = Linear(
            params.dim, params.vocab_size, bias=False
        )

        self.freqs_cis = precompute_freqs_cis(
            self.params.dim // self.params.n_heads, self.params.max_seq_len * 2
        )

        self.backbone = clip.load('ViT-L/14')[0]
        self.adapter_emb1 = nn.Parameter(torch.randn(1, params.emb, params.dim) * 0.02)
        self.adapter_emb2 = nn.Parameter(torch.zeros(1, params.emb, params.dim))

        self.adapter_proj = Projector(1024, params.hidden_proj, params.dim).float()
        self.adapter_modality_embedding=nn.Embedding(2,params.dim).float()
    def forward(self, examples, labels, images=None, prefix_img=None, prefix_nonimg=None, img_indicators=None):

        image_embeds = self.backbone.encode_image(images).half()
        if isinstance(img_indicators, list):
            img_indicators = torch.Tensor(img_indicators).to(image_embeds.device).long()
        
        image_embeds = self.adapter_proj(image_embeds) * 0.01
        image_embeds = image_embeds * img_indicators.half().view(-1, 1, 1)
        image_embeds = torch.cat([image_embeds, torch.zeros(image_embeds.size(0), self.adapter_emb1.size(1) - 256,
                                                            image_embeds.size(2)).half().to(image_embeds.device)],
                                 dim=1)
        _bsz, seqlen = examples.shape

        examples = self.tok_embeddings(examples)

        h = examples

        seqlen = (labels > 0).float().nonzero(as_tuple=False)[:, 1].max() + 1
        h = h[:, :seqlen]
        labels = labels[:, :seqlen]
        seqlen = h.size(1)
        freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = freqs_cis[:seqlen]
        mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=h.device)
        mask = torch.triu(mask, diagonal=0 + 1).type_as(h)

        start_pos = 0
        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask, [image_embeds, self.adapter_emb1, self.adapter_emb2])
        h = self.norm(h)
        output = self.output(h)
        output = output[:, :-1, :].reshape(-1, self.vocab_size)
        labels = labels[:, 1:].flatten()

        c_loss = self.criterion(output, labels)
        return c_loss

    @torch.inference_mode()
    def generate(
            self,
            prompts,
            images,
            indicators,
            max_gen_len,
            tokenizer = None,
            temperature: float = 0,
            top_p: float = 0.95,
    ):
        bsz = len(prompts)
        params = self.params
        self.eval()

        images = images.cuda()
        self.backbone.cuda()

        image_embeds = self.backbone.encode_image(images)
        image_embeds = self.adapter_proj(image_embeds) * 0.01

        prompt_tokens = []
        for i, x in enumerate(prompts):
            token_idx = tokenizer.encode(x, bos=True, eos=False)
            prompt_tokens.append(token_idx)

        max_prompt_size = max([len(t) for t in prompt_tokens])
        prompt_size = [len(t) for t in prompt_tokens]
        total_len = min(512, max_gen_len + max_prompt_size)

        tokens = torch.full((bsz, total_len), 0).cuda().long()
        mask = torch.full((bsz, 1, total_len, total_len), float("-inf"), device=tokens.device)
        mask = torch.triu(mask, diagonal=1)

        for k, t in enumerate(prompt_tokens):
            t = t[:total_len - max_gen_len]
            tokens[k, -len(t) - max_gen_len:- max_gen_len] = torch.tensor(t).long()
            mask[k, :, -len(t) - max_gen_len:, :-len(t) - max_gen_len] = float("-inf")

        token_embeds = self.tok_embeddings(tokens)
        indicators = torch.Tensor(indicators).cuda().long()
        image_embeds = image_embeds * indicators.half().view(-1, 1, 1)
        image_embeds = torch.cat([image_embeds, torch.zeros(image_embeds.size(0), self.adapter_emb1.size(1) - 256,
                                                            image_embeds.size(2)).half().to(image_embeds.device)], dim=1)

        mask = mask.type_as(token_embeds)

        start_pos = min(max_prompt_size, 512 - max_gen_len)
        stop_flag = torch.ones([bsz], dtype=torch.long).cuda()


        prev_pos = 0
        for cur_pos in range(start_pos, total_len):
            h = token_embeds[:, prev_pos:cur_pos]
            
            mask_input = mask[:, :, prev_pos:cur_pos, :cur_pos]

            with autocast():
                _bsz, seqlen, _ = h.shape
                self.freqs_cis = self.freqs_cis.to(h.device)
                freqs_cis = self.freqs_cis[prev_pos: prev_pos + seqlen]

                if mask_input is None and seqlen > 1:
                    mask_input = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=tokens.device)
                    mask_input = torch.triu(mask_input, diagonal=prev_pos + 1).type_as(h)

                for layer in self.layers:
                    h = layer(h, prev_pos, freqs_cis, mask_input, [image_embeds, self.adapter_emb1, self.adapter_emb2])

                h = self.norm(h)
                output = self.output(h[:, -1, :])  # only compute last logits
                logits = output.float()

            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits, dim=-1)
            next_token = next_token.reshape(-1)
            stop_flag *= (next_token != tokenizer.eos_id).long()
            if stop_flag.sum() == 0:
                tokens[:, cur_pos] = next_token
                break

            next_token_embeds = self.tok_embeddings(next_token)

            token_embeds[:, cur_pos] = next_token_embeds
            tokens[:, cur_pos] = next_token

            prev_pos = cur_pos

        decoded = []
        for i, t in enumerate(tokens.tolist()):
            try:
                t = t[- max_gen_len:]
                t = t[: t.index(tokenizer.eos_id)]
            except ValueError:
                pass
            decoded.append(tokenizer.decode(t))

        return decoded


def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token
