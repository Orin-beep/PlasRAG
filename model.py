from einops import rearrange, repeat
from einops_exts import rearrange_many
from transformers import PreTrainedModel, PretrainedConfig, AutoConfig
import numpy as np
from transformers import BertPreTrainedModel    #, BertModel
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask_for_sdpa
from dataclasses import dataclass
#from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.file_utils import ModelOutput
from typing import List, Optional, Tuple
from transformers.models.bert.modeling_bert import BertEmbeddings, BertEncoder
from typing import List, Optional, Tuple, Union
import pickle as pkl
from torch.nn import BCEWithLogitsLoss
import sys
import time
import torch
from torch import nn, einsum


@dataclass
class SequenceClassifierOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    embed: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class ProteinTokenizerConfig(PretrainedConfig): # transform protein embeddings to protein tokens
    def __init__(
        self,
        prot_feature_dim: int = 768,   # map ESM embeddings from 1280 to 768
        bert_embedding_dim: int = 768,
        num_prot_tokens: int = 32,
        **kwargs,
    ):
        self.prot_feature_dim = prot_feature_dim
        self.bert_embedding_dim = bert_embedding_dim
        self.num_prot_tokens = num_prot_tokens
        super().__init__(**kwargs)


class retrieverConfig(PretrainedConfig):
    def __init__(
        self,
        **kwargs,
    ):
        self.prot_tokenizer_config = ProteinTokenizerConfig(**{})
        self.pretrained_bert = None
        super().__init__(**kwargs)


def FeedForward(dim, mult=4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )


class PerceiverAttention(nn.Module):
    def __init__(self, *, dim, dim_head=64, heads=12):   # 12 heads, 64 dim -> 768
        super().__init__()

        self.scale = dim_head**-0.5
        self.heads = heads
        inner_dim = dim_head * heads    # 768

        self.norm_media = nn.LayerNorm(dim)
        self.norm_latents = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, latents, protein_attn_masks):
        x = self.norm_media(x)
        latents = self.norm_latents(latents)

        h = self.heads
        q = self.to_q(latents)  # [B, 1, 16, 768]
        kv_input = torch.cat((x, latents), dim=-2)  # [B, 1, 316, 768]

        protein_attn_masks = torch.cat((protein_attn_masks,
                    torch.ones(
                        (latents.shape[0], latents.shape[-2]),
                        dtype=latents.dtype,
                        device=latents.device,),),dim=-1,)

        k, v = self.to_kv(kv_input).chunk(2, dim=-1)    # [3, 1, 364, 1024]
        q, k, v = rearrange_many((q, k, v), "b t n (h d) -> b h t n d", h=h)
        q = q * self.scale
       
        # attention
        sim = einsum("... i d, ... j d  -> ... i j", q, k)  # [3, 16, 1, 64, 364], cross-attention
        
        # masking
        attn_bias = torch.zeros(
            (q.size(0), 1, 1, q.size(-2), k.size(-2)),
            dtype=q.dtype,
            device=q.device,)
        protein_attn_masks = repeat(
            protein_attn_masks, "b n -> b 1 1 l n", l=q.size(-2))
        attn_bias.masked_fill_(protein_attn_masks.logical_not(), float("-inf"))
        sim += attn_bias

        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)
        out = einsum("... i j, ... j d -> ... i d", attn, v)
        out = rearrange(out, "b h t n d -> b t n (h d)", h=h)
        return self.to_out(out)


class ProteinTokenizer(PreTrainedModel):
    def __init__(self, config: ProteinTokenizerConfig):
        super().__init__(config)
        dim = config.prot_feature_dim          # 768
        dim_inner = config.bert_embedding_dim  # 768
        num_latents = config.num_prot_tokens        # 24
        depth = 1
        dim_head = 64
        heads=12    # dim of transformer: 64*16 = 1024 
        ff_mult=4

        #self.projection = nn.Linear(dim, dim_inner) # 1024 -> 768
        self.latents = nn.Parameter(torch.randn(num_latents, dim))
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                    FeedForward(dim=dim, mult=ff_mult),]))
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, protein_attn_masks):
        b, T, F, v = x.shape[:4]    # [3, 1, 1, 300, 1024]
        x = rearrange(
            x, "b T F v d -> b T (F v) d"
        )  # flatten the frame and spatial dimensions

        latents = self.latents
        latents = repeat(latents, "n d -> b T n d", b=b, T=T)   # [3, 1, 300/64, 1024]
        
        for attn, ff in self.layers:
            latents = attn(x, latents, protein_attn_masks) + latents
            latents = ff(latents) + latents

        return self.norm(latents)
        #return self.projection(self.norm(latents))


class retriever(PreTrainedModel):
    def __init__(self, config:retrieverConfig, raw_embed, tokenizer, vocab, max_length, text_embed_path, device, domain):
        super().__init__(config)
        self.raw_embed = raw_embed
        self.raw_embed_dim = 1280
        self.fc = torch.nn.Linear(self.raw_embed_dim, config.prot_tokenizer_config.prot_feature_dim)
        self.text_embed_path = text_embed_path
        self.text_embeddings = torch.load(self.text_embed_path).to(device)
        self.DEV = device
        config.prot_tokenizer_config.num_prot_tokens = vocab#len(vocab)
        self.bert = None
        
        self.num_prot_tokens = config.prot_tokenizer_config.num_prot_tokens
        self.prot_tokenizer = ProteinTokenizer(config.prot_tokenizer_config)
       
        self.transformer_width = 768
        self.text_projection = nn.Parameter(torch.empty(self.transformer_width, 512))
        self.plas_projection = nn.Parameter(torch.empty(self.transformer_width, 512))
        nn.init.normal_(self.text_projection, std=self.transformer_width ** -0.5)
        nn.init.normal_(self.plas_projection, std=self.transformer_width ** -0.5)

        self.logit_scale = nn.Parameter(torch.ones([vocab]) * np.log(1 / 0.07))
        self.post_init()    # Initialize weights and apply final processing

    def forward(
        self,
        proteins: torch.Tensor,
        labels: Optional[torch.Tensor] = None,):
       
        # plasmid embeddings (768)
        protein_features, protein_attn_masks = self.assemble_features(proteins)
        protein_features = self.fc(protein_features)
        protein_tokens = self.prot_tokenizer(protein_features, protein_attn_masks)
        B = proteins.shape[0]
        protein_tokens = protein_tokens.view(B, self.num_prot_tokens, -1)   # [5, 32, 768]
        device = self.DEV
        
        last_hidden_state = self.text_embeddings@self.text_projection
        protein_tokens = protein_tokens@self.plas_projection

        # cosine similarity
        protein_tokens = protein_tokens / protein_tokens.norm(dim=2, keepdim=True)
        last_hidden_state = last_hidden_state / last_hidden_state.norm(dim=1, keepdim=True)
        logit_scale = self.logit_scale.exp()    # init: 14
        logits = torch.einsum('ijk,jk->ij', protein_tokens, last_hidden_state)
        logits = logits * logit_scale
      
        probs = torch.sigmoid(logits)
        probs = probs/probs.sum(dim=1, keepdim=True)
        embed = torch.bmm(probs.unsqueeze(1), protein_tokens).squeeze(1)   # [2048, 1, 124] * [2048, 124, 512]
        # calculate BCE loss with ignoring -100
        #loss_fct = BCEWithLogitsLoss()
        #mask = labels.ne(-100)
        #predictions = logits[mask]
        #labels = labels[mask]
        #loss = loss_fct(predictions, labels)
        
        return SequenceClassifierOutput(
            #loss=loss,
            embed=embed,
            logits=logits,)

    def assemble_features(self, proteins):
        batch_size, max_len, device = proteins.shape[0], proteins.shape[1], proteins.device
        protein_attn_masks = (proteins != 0).int()
        proteins = proteins.cpu()
        protein_features = self.raw_embed[proteins]
        protein_features = protein_features.to(device)
        protein_features = protein_features.view(batch_size, 1, 1, max_len, self.raw_embed_dim)

        return protein_features, protein_attn_masks


def get_vram():
    free = torch.cuda.mem_get_info()[0] / 1024 ** 3
    total = torch.cuda.mem_get_info()[1] / 1024 ** 3
    total_cubes = 24
    free_cubes = int(total_cubes * free / total)
    print(f'VRAM: {total - free:.2f}/{total:.2f}GB\t VRAM:[' + (total_cubes - free_cubes) * '▮' + free_cubes * '▯' + ']')

