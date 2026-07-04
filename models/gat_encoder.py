import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from models.model_utils import index_select_ND


class GATEncoder(nn.Module):
    """Class: 'MPNEncoder' is a message passing neural network for encoding molecules."""

    def __init__(self, atom_fdim: int, bond_fdim: int, hidden_size: int,
                 depth: int, num_heads: int = 1, dropout: float = 0.15, atom_message: bool = False,
                 direction_mode: str = 'none',
                 alpha_input: str = 'none',
                 alpha_apply_mode: str = 'none'):
        """
        Parameters
        ----------
        atom_fdim: Atom feature vector dimension.
        bond_fdim: Bond feature vector dimension.
        hidden_size: Hidden layers dimension
        depth: Number of message passing steps
        droupout: the droupout rate
        atom_message: 'D-MPNN' or 'MPNN', centers messages on bonds or atoms.
        """
        super(GATEncoder, self).__init__()
        self.atom_fdim = atom_fdim
        self.bond_fdim = bond_fdim
        self.hidden_size = hidden_size
        self.depth = depth
        self.dropout = dropout
        self.num_heads = num_heads
        self.alpha_input = alpha_input  # Ablation knobs: 'none' | 'el' | 'e0_el' | 'scalar_gate_e0_el' | 'vector_gate_e0_el'
        self.alpha_apply_mode = alpha_apply_mode # alpha apply: 'none' | 'h_attn' | 'm_attn' | 'out_attn'| 'm_out_attn'

        self.w_a = nn.Linear(self.atom_fdim, self.hidden_size, bias=False) # 原子隐藏表示的线性转换层
        self.w_b = nn.Linear(self.bond_fdim, self.hidden_size, bias=False) # 边隐藏表示的线性转换层
        self.gru = nn.GRUCell(self.hidden_size, self.hidden_size) # 门控循环单元
        self.W_o = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size), nn.ReLU()) # 中间层原子隐藏表示的线性转换层
        self.W_p = nn.Sequential(nn.Linear(self.atom_fdim + self.hidden_size, self.hidden_size), nn.ReLU()) # 最终层输出原子隐藏表示的线性转换层
        self.dropout_layer = nn.Dropout(p=self.dropout) # 丢弃层

        if self.alpha_input == 'scalar_gate_e0_el':
            self.gate_b = nn.Linear(2 * self.hidden_size, 1) # 标量门控
            self.ln_b = nn.LayerNorm(self.hidden_size) # 层归一化
        if self.alpha_input == 'vector_gate_e0_el':
            self.gate_net_b = nn.Linear(2 * self.hidden_size, self.hidden_size) # 向量门控
            self.ln_b = nn.LayerNorm(self.hidden_size) # 层归一化

        if self.alpha_input == 'e0_el':
            self.att_vector = nn.Parameter(torch.randn(self.num_heads, self.hidden_size * 4)) # 注意力向量
        else:
            self.att_vector = nn.Parameter(torch.randn(self.num_heads, self.hidden_size * 3))
        nn.init.xavier_uniform_(self.att_vector) # 初始化注意力向量


    def forward(self, graph_tensors: Tuple[torch.Tensor], mask: torch.Tensor) -> torch.FloatTensor:
        """
        Forward pass of the graph encoder. Encodes a batch of molecular graphs.

        Parameters
        ----------
        graph_tensors: Tuple[torch.Tensor],
            Tuple of graph tensors - Contains atom features, message vector details, the incoming bond indices of atoms
            the index of the atom the bond is coming from, the index of the reverse bond and the undirected bond index 
            to the beginindex and endindex of the atoms.
        mask: torch.Tensor,
            Masks on nodes
        """
        f_atoms, f_bonds, a2b, b2a, b2revb, undirected_b2a = graph_tensors

        input_b = self.w_b(f_bonds) # 初始键特征: num_bonds x hidden_size

        message = input_b # 初始消息: num_bonds x hidden_size
        message_mask = torch.ones(message.size(0), 1, device=message.device) # 消息掩码: num_bonds x 1
        message_mask[0, 0] = 0  # 第一个消息是0填充

        num_atoms = f_atoms.size(0) # 原子数

        input_a = self.w_a(f_atoms) # 原子隐藏表示的线性转换层: num_atoms x hidden_size
        prev_atom_hiddens = input_a # 前一层原子隐藏状态: num_atoms x hidden_size

        src_atoms = b2a # 每条边的源原子（bond -> source atom）
        dst_atoms = b2a[b2revb] # 每条边的目标原子（bond -> destination atom）

        for depth in range(self.depth):
            prev_bond_hidden = message

            # 计算α权重值的注意力机制
            h_i_prev = prev_atom_hiddens[src_atoms] # num_bonds x hidden
            h_j_prev = prev_atom_hiddens[dst_atoms] # num_bonds x hidden

            if self.alpha_input == 'scalar_gate_e0_el':
                lam = torch.sigmoid(self.gate_b(torch.cat([input_b, prev_bond_hidden], dim=-1)))
                fused = lam * input_b + (1 - lam) * prev_bond_hidden
                fused = self.dropout_layer(fused)
                u = self.ln_b(fused)
                z_ij = torch.cat([h_i_prev, h_j_prev, u], dim=1)  # num_bonds x 3H
            elif self.alpha_input == 'vector_gate_e0_el':
                gate = torch.sigmoid(self.gate_net_b(torch.cat([input_b, prev_bond_hidden], dim=-1)))  # [B, E, D]
                fused = gate * prev_bond_hidden + (1.0 - gate) * input_b
                fused = self.dropout_layer(fused)
                u = self.ln_b(fused)
                z_ij = torch.cat([h_i_prev, h_j_prev, u], dim=1)  # num_bonds x 3H
            elif self.alpha_input == 'el':
                z_ij = torch.cat([h_i_prev, h_j_prev, prev_bond_hidden], dim=1)  # num_bonds x 3H
            elif self.alpha_input == 'e0_el':
                z_ij = torch.cat([h_i_prev, h_j_prev, input_b, prev_bond_hidden], dim=1)  # num_bonds x 4H
            else:
                z_ij = torch.cat([h_i_prev, h_j_prev, input_b], dim=1)  # num_bonds x 3H
            
            logits = F.leaky_relu(F.linear(z_ij, self.att_vector), negative_slope=0.2) # num_heads x num_bonds
            exp_c = torch.exp(logits)  # [num_bonds, num_heads]
            denom = torch.zeros(num_atoms, self.num_heads, device=exp_c.device, dtype=exp_c.dtype)

            if self.alpha_apply_mode == 'h_attn':
                denom.index_add_(0, src_atoms, exp_c)  # 聚合到每个源节点 [num_atoms, num_heads]
                norm = exp_c / (denom[src_atoms] + 1e-8)  # [num_bonds, num_heads]
            else:
                denom.index_add_(0, dst_atoms, exp_c)  # 聚合到每个目标节点 [num_atoms, num_heads]
                norm = exp_c / (denom[dst_atoms] + 1e-8)  # [num_bonds, num_heads]

            alpha = norm.mean(dim=1, keepdim=True)  # 平均多头结果：[num_bonds, 1]

            # 计算消息
            nei_a_message = index_select_ND(prev_bond_hidden, a2b)

            if self.alpha_apply_mode in ['m_attn', 'm_out_attn']:
                nei_alpha = index_select_ND(alpha, a2b) # num_atoms x max_num_bonds x hidden
                a_message = (nei_alpha * nei_a_message).sum(dim=1)  # num_atoms x hidden
            else:
                a_message = nei_a_message.sum(dim=1)  # num_atoms x hidden

            rev_message = prev_bond_hidden[b2revb]  # num_bonds x hidden
            m = a_message[b2a] - rev_message

            # GRU门控更新
            input_attn = alpha * input_b
            if self.alpha_apply_mode == 'h_attn':
                message = self.gru(input_attn, m)  # num_bonds x hidden_size
            else:
                message = self.gru(input_b, m)  # num_bonds x hidden_size

            message = message * message_mask
            message = self.dropout_layer(message)  # num_bonds x hidden
            
            # 为下一层更新 h^{(l-1)}
            nei_a_message = index_select_ND(message, a2b)

            if self.alpha_apply_mode in ['out_attn', 'm_out_attn']:
                nei_alpha = index_select_ND(alpha, a2b) # num_atoms x max_num_bonds x hidden
                a_message = (nei_alpha * nei_a_message).sum(dim=1)  # num_atoms x hidden
            else:
                a_message = nei_a_message.sum(dim=1)  # num_atoms x hidden

            if depth != self.depth - 1:
                atom_hiddens = self.W_o(a_message)
            else:
                a_input = torch.cat([f_atoms, a_message], dim=1)
                atom_hiddens = self.W_p(a_input)  # num_atoms x hidden

            prev_atom_hiddens = atom_hiddens

        return atom_hiddens * mask
