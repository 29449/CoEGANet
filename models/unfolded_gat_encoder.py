import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from models.model_utils import index_select_ND

class UnfoldedGATEncoder(nn.Module):
    def __init__(self, atom_fdim: int, bond_fdim: int, hidden_size: int,
                 depth: int, num_heads: int = 1, dropout: float = 0.15, 
                 atom_message: bool = False, direction_mode: str = 'none',
                 alpha_input: str = 'none', alpha_apply_mode: str = 'none'):
        super(UnfoldedGATEncoder, self).__init__()
        self.atom_fdim = atom_fdim
        self.bond_fdim = bond_fdim
        self.hidden_size = hidden_size
        self.depth = depth
        self.dropout = dropout
        self.num_heads = num_heads
        self.alpha_input = alpha_input
        self.alpha_apply_mode = alpha_apply_mode

        # 初始化层（保持不变）
        self.w_a = nn.Linear(self.atom_fdim, self.hidden_size, bias=False)
        self.w_b = nn.Linear(self.bond_fdim, self.hidden_size, bias=False)
        self.dropout_layer = nn.Dropout(p=self.dropout)

        # 关键修改：为每一层创建独立的参数
        self.layers = nn.ModuleList()
        for i in range(depth):
            layer = UnfoldedGATLayer(
                hidden_size=hidden_size,
                num_heads=num_heads,
                dropout=dropout,
                alpha_input=alpha_input,
                alpha_apply_mode=alpha_apply_mode,
                is_final_layer=(i == depth-1),
                atom_fdim=atom_fdim
            )
            self.layers.append(layer)

    def forward(self, graph_tensors: Tuple[torch.Tensor], mask: torch.Tensor) -> torch.FloatTensor:
        f_atoms, f_bonds, a2b, b2a, b2revb, undirected_b2a = graph_tensors

        # 初始化（保持不变）
        input_b = self.w_b(f_bonds)
        message = input_b
        message_mask = torch.ones(message.size(0), 1, device=message.device)
        message_mask[0, 0] = 0

        input_a = self.w_a(f_atoms)
        prev_atom_hiddens = input_a

        src_atoms = b2a
        dst_atoms = b2a[b2revb]

        # 关键修改：逐层前向传播，而不是循环
        for layer_idx, layer in enumerate(self.layers):
            message, prev_atom_hiddens = layer(
                layer_idx=layer_idx,
                f_atoms=f_atoms,
                input_b=input_b,
                message=message,
                prev_atom_hiddens=prev_atom_hiddens,
                src_atoms=src_atoms,
                dst_atoms=dst_atoms,
                a2b=a2b,
                b2a=b2a,
                b2revb=b2revb,
                message_mask=message_mask,
                num_atoms=f_atoms.size(0)
            )

        if mask is None:
            mask = torch.ones(prev_atom_hiddens.size(0), 1, device=f_atoms.device)
            mask[0, 0] = 0

        return prev_atom_hiddens * mask

class UnfoldedGATLayer(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout: float,
                 alpha_input: str, alpha_apply_mode: str, is_final_layer: bool,
                 atom_fdim: int):
        super(UnfoldedGATLayer, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.alpha_input = alpha_input
        self.alpha_apply_mode = alpha_apply_mode
        self.is_final_layer = is_final_layer
        
        # 关键：每层有自己的GRU和线性变换
        self.gru = nn.GRUCell(self.hidden_size, self.hidden_size)
        
        # 每层有自己的输出变换
        if not self.is_final_layer:
            self.W_o = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size), 
                nn.ReLU()
            )
        else:
            self.W_p = nn.Sequential(
                nn.Linear(atom_fdim + self.hidden_size, self.hidden_size), 
                nn.ReLU()
            )
        
        self.dropout_layer = nn.Dropout(p=dropout)

        # 每层有自己的门控参数（如果使用）
        if self.alpha_input == 'scalar_gate_e0_el':
            self.gate_b = nn.Linear(2 * self.hidden_size, 1)
            self.ln_b = nn.LayerNorm(self.hidden_size)
        if self.alpha_input == 'vector_gate_e0_el':
            self.gate_net_b = nn.Linear(2 * self.hidden_size, self.hidden_size) # 向量门控
            self.ln_b = nn.LayerNorm(self.hidden_size)
            # self.gru_b = nn.GRUCell(self.hidden_size, self.hidden_size)

        # 每层有自己的注意力参数
        if self.alpha_input == 'e0_el':
            self.att_vector = nn.Parameter(torch.randn(self.num_heads, self.hidden_size * 4))
        else:
            self.att_vector = nn.Parameter(torch.randn(self.num_heads, self.hidden_size * 3))
        nn.init.xavier_uniform_(self.att_vector)

    def forward(self, layer_idx: int, f_atoms: torch.Tensor, input_b: torch.Tensor, 
                message: torch.Tensor, prev_atom_hiddens: torch.Tensor,
                src_atoms: torch.Tensor, dst_atoms: torch.Tensor,
                a2b: torch.Tensor, b2a: torch.Tensor, b2revb: torch.Tensor,
                message_mask: torch.Tensor, num_atoms: int):
        
        prev_bond_hidden = message

        # 计算注意力权重（与原代码类似，但使用本层的参数）
        h_i_prev = prev_atom_hiddens[src_atoms]
        h_j_prev = prev_atom_hiddens[dst_atoms]

        if self.alpha_input == 'scalar_gate_e0_el':
            lam = torch.sigmoid(self.gate_b(torch.cat([input_b, prev_bond_hidden], dim=-1)))
            fused = lam * input_b + (1 - lam) * prev_bond_hidden
            fused = self.dropout_layer(fused)
            u = self.ln_b(fused)
            z_ij = torch.cat([h_i_prev, h_j_prev, u], dim=1)
        elif self.alpha_input == 'vector_gate_e0_el':
            gate = torch.sigmoid(self.gate_net_b(torch.cat([input_b, prev_bond_hidden], dim=-1)))  # [B, E, D]
            fused = gate * prev_bond_hidden + (1.0 - gate) * input_b
            # fused = self.gru_b(input_b, prev_bond_hidden)
            fused = self.dropout_layer(fused)
            u = self.ln_b(fused)
            z_ij = torch.cat([h_i_prev, h_j_prev, u], dim=1)
        elif self.alpha_input == 'el':
            z_ij = torch.cat([h_i_prev, h_j_prev, prev_bond_hidden], dim=1)
        elif self.alpha_input == 'e0_el':
            z_ij = torch.cat([h_i_prev, h_j_prev, input_b, prev_bond_hidden], dim=1)
        else:
            z_ij = torch.cat([h_i_prev, h_j_prev, input_b], dim=1)
        
        # 使用本层的注意力参数
        logits = F.leaky_relu(F.linear(z_ij, self.att_vector), negative_slope=0.2)
        logits = logits - logits.max(dim=1, keepdim=True)[0]
        exp_c = torch.exp(logits)
        denom = torch.zeros(num_atoms, self.num_heads, device=exp_c.device, dtype=exp_c.dtype)

        if self.alpha_apply_mode == 'h_attn':
            denom.index_add_(0, src_atoms, exp_c)
            norm = exp_c / (denom[src_atoms] + 1e-8)
        else:
            denom.index_add_(0, dst_atoms, exp_c)
            norm = exp_c / (denom[dst_atoms] + 1e-8)

        alpha = norm.mean(dim=1, keepdim=True)

        # 计算消息
        nei_a_message = index_select_ND(prev_bond_hidden, a2b)

        if self.alpha_apply_mode == 'm_attn':
            nei_alpha = index_select_ND(alpha, a2b)
            a_message = (nei_alpha * nei_a_message).sum(dim=1)
        else:
            a_message = nei_a_message.sum(dim=1)

        rev_message = prev_bond_hidden[b2revb]
        m = a_message[b2a] - rev_message

        # 使用本层的GRU
        input_attn = alpha * input_b
        if self.alpha_apply_mode == 'h_attn':
            message = self.gru(input_attn, m)
        else:
            message = self.gru(input_b, m)
        
        message = message * message_mask
        message = self.dropout_layer(message)

        # 更新原子隐藏状态
        nei_a_message = index_select_ND(message, a2b)

        if self.alpha_apply_mode == 'out_attn':
            nei_alpha = index_select_ND(alpha, a2b)
            a_message = (nei_alpha * nei_a_message).sum(dim=1)
        else:
            a_message = nei_a_message.sum(dim=1)

        # 使用本层的输出变换
        if not self.is_final_layer:
            atom_hiddens = self.W_o(a_message)
        else:
            a_input = torch.cat([f_atoms, a_message], dim=1)
            atom_hiddens = self.W_p(a_input)

        return message, atom_hiddens
        