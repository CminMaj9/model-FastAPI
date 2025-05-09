import torch
import torch.nn as nn
import torch.nn.functional as F

# ===== GAT图注意力层 =====
class GraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.0, alpha=0.2):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a = nn.Parameter(torch.empty(size=(2 * out_dim, 1)))
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def forward(self, h, adj):
        Wh = self.W(h)  # [B, N, F']
        B, N, F_ = Wh.shape
        a_input = torch.cat([
            Wh.unsqueeze(2).repeat(1, 1, N, 1),
            Wh.unsqueeze(1).repeat(1, N, 1, 1)
        ], dim=-1)  # [B, N, N, 2F']

        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(-1))  # [B, N, N]
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=-1)
        attention = self.dropout(attention)

        h_prime = torch.matmul(attention, Wh)  # [B, N, F']
        return h_prime

# ===== 多层 Transformer 编码器模块 =====
class TemporalTransformerBlock(nn.Module):
    def __init__(self, embed_dim, heads, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        ff_out = self.ffn(x)
        x = self.norm2(x + ff_out)
        return x

# ===== 主模型结构 v7 =====
class GAT_Transformer_ContextFusion(nn.Module):
    def __init__(self, num_nodes, in_channels, gat_hidden, output_dim, seq_len,
                 context_dim, node_emb_dim=16, heads=4, num_layers=2):
        super().__init__()
        self.num_nodes = num_nodes
        self.seq_len = seq_len
        self.context_dim = context_dim
        self.node_emb_dim = node_emb_dim
        self.gat_hidden = gat_hidden
        self.final_embed_dim = gat_hidden + node_emb_dim

        self.node_emb = nn.Embedding(num_nodes, node_emb_dim)
        self.gat = GraphAttentionLayer(in_channels, gat_hidden)

        self.temporal_blocks = nn.ModuleList([
            TemporalTransformerBlock(self.final_embed_dim, heads) for _ in range(num_layers)
        ])

        self.context_mlp = nn.Sequential(
            nn.Linear(seq_len * context_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, self.final_embed_dim),
            nn.Sigmoid()
        )

        self.head = nn.Sequential(
            nn.Linear(self.final_embed_dim, self.final_embed_dim),
            nn.ReLU(),
            nn.LayerNorm(self.final_embed_dim),
            nn.Linear(self.final_embed_dim, output_dim),
            nn.Sigmoid()  # ✅ 限制输出在 0~1
        )

    def forward(self, x, adj, context):
        B, N, T, _ = x.shape
        node_ids = torch.arange(N, device=x.device).unsqueeze(0).repeat(B, 1)
        node_embed = self.node_emb(node_ids)

        outputs = []
        for t in range(T):
            xt = x[:, :, t, :]
            out = torch.relu(self.gat(xt, adj))
            outputs.append(out)

        gcn_seq = torch.stack(outputs, dim=2)  # [B, N, T, H]
        gcn_seq = gcn_seq.permute(0, 2, 1, 3).reshape(B * N, T, -1)

        node_embed_seq = node_embed.unsqueeze(2).repeat(1, 1, T, 1).reshape(B * N, T, -1)
        gcn_seq = torch.cat([gcn_seq, node_embed_seq], dim=-1)  # [B*N, T, D]

        x = gcn_seq
        for block in self.temporal_blocks:
            x = block(x)

        last_out = x[:, -1, :]  # [B*N, D]
        context = context.view(B * N, T * self.context_dim)
        context_weight = self.context_mlp(context)

        fused = last_out * context_weight
        pred = self.head(fused).view(B, N, -1)
        return pred