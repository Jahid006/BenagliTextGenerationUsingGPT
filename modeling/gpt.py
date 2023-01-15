import math
import torch
from torch import nn
import torch.nn.functional as F


class MultiheadAttention(nn.Module):
    def __init__(self, config):
        super(MultiheadAttention, self).__init__()
        embed_dim = config.embed_dim
        self.num_heads = config.num_heads
        assert embed_dim % self.num_heads == 0, (
            "invalid heads and embedding dimension configuration"
        )

        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(config.attn_dropout)
        self.proj_dropout = nn.Dropout(config.ff_dropout)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.max_len, config.max_len))
            .unsqueeze(0).unsqueeze(0)
        )
    
    def forward(self, x):
        batch_size = x.size(0)
        seq_len = x.size(1)
        # x.shape == (batch_size, seq_len, embed_dim)
        k_t = self.key(x).reshape(batch_size, seq_len, self.num_heads, -1).permute(0,2,3,1)#.transpose(1, 2)
        v = self.value(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        q = self.query(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        # shape == (batch_size, num_heads, seq_len, head_dim)
        #k_t = k_t.transpose(3, 2)
        
        attn = torch.matmul(q, k_t) * (1.0 / math.sqrt(q.size(-1)))
        # attn.shape == (batch_size, num_heads, seq_len, seq_len)
        mask = self.mask[:, :, :seq_len, :seq_len]
        attn = attn.masked_fill(mask == 0, float("-inf"))
        # attn.shape == (batch_size, num_heads, seq_len, seq_len)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        y = torch.matmul(attn, v)
        # y.shape == (batch_size, num_heads, seq_len, head_dim)
        y = y.transpose(1, 2).contiguous()
        # y.shape == (batch_size, seq_len, num_heads, head_dim)
        y = y.reshape(batch_size, seq_len, -1)
        # y.shape == (batch_size, seq_len, embed_dim)
        y = self.proj_dropout(self.proj(y))
        return y


class Block(nn.Module):
    def __init__(self, config):
        super(Block, self).__init__()
        embed_dim = config.embed_dim
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.attn = MultiheadAttention(config)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(config.ff_dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class GPTPretrained(nn.Module):
    def __init__(self, config):
        super(GPTPretrained, self).__init__()
        
        self.max_len = config.max_len
        self.tok_embed = nn.Embedding(
            config.vocab_size, config.embed_dim
        )
        self.pos_embed = nn.Parameter(
            torch.zeros(1, config.max_len, config.embed_dim)
        )
        self.dropout = nn.Dropout(config.embed_dropout)
        self.blocks = nn.ModuleList(
            [Block(config) for _ in range(config.num_blocks)]
        )
        
    def forward(self, x, target=None):
        # batch_size = x.size(0)
        seq_len = x.size(1)
        assert seq_len <= self.max_len, "sequence longer than model capacity"
        
        tok_embedding = self.tok_embed(x)
        # tok_embedding.shape == (batch_size, seq_len, embed_dim)
        pos_embedding = self.pos_embed[:, :seq_len, :]
        # pos_embedding.shape == (1, seq_len, embed_dim)
        x = self.dropout(tok_embedding + pos_embedding)

        for block in self.blocks:
            x = block(x)
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super(GPT, self).__init__()
        self.encoder = GPTPretrained(config)
        self.ln = nn.LayerNorm(config.embed_dim)
        self.fc = nn.Linear(config.embed_dim, config.vocab_size)

    def forward(self, x, target=None):
        x = self.encoder(x, target)
        x = self.ln(x)
        x = self.fc(x)

        return x
