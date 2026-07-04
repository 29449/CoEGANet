"""
反应中心预测头模块
基于 Retro-MTGR 的思想，预测产物中需要断裂的键（反应中心）

支持 Retro-MTGR 风格的完整键特征构建，适用于 hidden_size=256 的情况
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional, Dict
from rdkit import Chem

from models.model_utils import index_select_ND


class ReactionCenterHead(nn.Module):
    """
    反应中心预测头：预测产物中每条键的断裂概率
    
    参考 Retro-MTGR 的 RedOut 模块设计，支持完整的键特征构建
    适用于 hidden_size=256 的情况
    """
    
    def __init__(self, hidden_size: int, bond_fdim: int = 4, 
                 use_bond_energy: bool = False,
                 use_atom_symbol: bool = False,
                 use_neighbor_bond_type: bool = False,
                 use_electronegativity: bool = False,
                 feature_mode: str = 'simple',
                 dropout: float = 0.15):
        """
        Parameters
        ----------
        hidden_size: int
            编码器输出的隐藏层维度（Graph2Edits-final 默认 256）
        bond_fdim: int
            键特征维度（键类型：单键/双键/三键/芳香键）
        use_bond_energy: bool
            是否使用键能特征（需要额外计算和键能表）
        use_atom_symbol: bool
            是否使用原子符号 one-hot（23维×2，会增加46维）
        use_neighbor_bond_type: bool
            是否使用邻居键类型特征（4维）
        use_electronegativity: bool
            是否使用电负性差特征（1维，需要电负性表）
        feature_mode: str
            'simple': 简化模式（键类型 + GCN特征 + 键能）
            'full': 完整模式（所有特征，类似 Retro-MTGR）
            'minimal': 最小模式（仅键类型 + GCN特征）
        dropout: float
            Dropout 比率
        """
        super(ReactionCenterHead, self).__init__()
        self.hidden_size = hidden_size
        self.bond_fdim = bond_fdim
        self.feature_mode = feature_mode
        
        # 根据 feature_mode 自动设置特征选项
        if feature_mode == 'full':
            use_atom_symbol = True
            use_neighbor_bond_type = True
            use_electronegativity = True
            use_bond_energy = True
        elif feature_mode == 'minimal':
            use_atom_symbol = False
            use_neighbor_bond_type = False
            use_electronegativity = False
            use_bond_energy = False
        
        self.use_bond_energy = use_bond_energy
        self.use_atom_symbol = use_atom_symbol
        self.use_neighbor_bond_type = use_neighbor_bond_type
        self.use_electronegativity = use_electronegativity
        
        # 原子符号 one-hot 维度（Retro-MTGR 使用 23 种原子）
        self.atom_symbol_dim = 23 if use_atom_symbol else 0
        
        # 键类型编码层
        self.bond_type_linear = nn.Linear(bond_fdim, bond_fdim)
        
        # 原子符号编码层（如果使用）
        if use_atom_symbol:
            self.atom_symbol_linear = nn.Linear(self.atom_symbol_dim, self.atom_symbol_dim)
        
        # 邻居键类型编码层（如果使用）
        if use_neighbor_bond_type:
            self.neighbor_bond_type_linear = nn.Linear(bond_fdim, bond_fdim)
        
        # 电负性差编码层（如果使用）
        if use_electronegativity:
            self.electronegativity_linear = nn.Linear(1, 1)
        
        # 键能编码层（如果使用）
        if use_bond_energy:
            self.bond_energy_linear = nn.Linear(1, 1)
        
        # 计算总特征维度
        # 基础：GCN 原子特征（相加） + 键类型
        feature_dim = hidden_size + bond_fdim
        
        # 如果使用原子符号，需要拼接而不是相加
        if use_atom_symbol:
            # Retro-MTGR 方式：(atom_symbol + GCN_feat) + (atom_symbol + GCN_feat)
            feature_dim = (self.atom_symbol_dim + hidden_size) * 2 + bond_fdim
        
        if use_neighbor_bond_type:
            feature_dim += bond_fdim
        if use_electronegativity:
            feature_dim += 1
        if use_bond_energy:
            feature_dim += 1
        
        # 投影层：如果特征维度太大，先投影到合理大小
        # 对于 hidden_size=256，完整特征约 567 维，投影到 256 或 512
        if feature_dim > 512:
            self.projection = nn.Linear(feature_dim, 512)
            predictor_input_dim = 512
        else:
            self.projection = None
            predictor_input_dim = feature_dim
        
        # 最终预测层
        self.predictor = nn.Sequential(
            nn.Linear(predictor_input_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, 1)
        )
        
    def get_bond_type_onehot(self, bond):
        """将 RDKit 键类型转换为 one-hot 编码"""
        bond_type = bond.GetBondType()
        if str(bond_type) == 'SINGLE':
            return torch.tensor([1, 0, 0, 0], dtype=torch.float32)
        elif str(bond_type) == 'DOUBLE':
            return torch.tensor([0, 1, 0, 0], dtype=torch.float32)
        elif str(bond_type) == 'TRIPLE':
            return torch.tensor([0, 0, 1, 0], dtype=torch.float32)
        elif str(bond_type) == 'AROMATIC':
            return torch.tensor([0, 0, 0, 1], dtype=torch.float32)
        else:
            return torch.tensor([0, 0, 0, 0], dtype=torch.float32)
    
    def get_atom_symbol_onehot(self, atom_symbol: str, device: torch.device) -> torch.Tensor:
        """
        获取原子符号的 one-hot 编码（23维，参考 Retro-MTGR）
        
        AtomSymbles = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 
                      'Ca', 'Fe', 'Al', 'I', 'B', 'K', 'Se', 'Zn', 'H', 'Cu', 'Mn', 
                      '*', 'unknown']
        """
        atom_symbols = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 
                       'Ca', 'Fe', 'Al', 'I', 'B', 'K', 'Se', 'Zn', 'H', 'Cu', 'Mn', 
                       '*', 'unknown']
        
        if atom_symbol not in atom_symbols:
            atom_symbol = 'unknown'
        
        idx = atom_symbols.index(atom_symbol)
        onehot = torch.zeros(self.atom_symbol_dim, device=device, dtype=torch.float32)
        if idx < self.atom_symbol_dim:
            onehot[idx] = 1.0
        return onehot
    
    def get_neighbor_bond_type(self, mol: Chem.Mol, atom_idx: int, device: torch.device) -> torch.Tensor:
        """
        计算原子所有邻居键类型的平均值（参考 Retro-MTGR 的 getbondneighbortype）
        
        Returns
        -------
        torch.Tensor: [bond_fdim] 邻居键类型的平均值
        """
        atomnum = mol.GetNumAtoms()
        bondtype_list = []
        count = 0
        
        for j in range(atomnum):
            bond = mol.GetBondBetweenAtoms(atom_idx, j)
            if bond is None:
                continue
            count += 1
            bondtype_list.append(self.get_bond_type_onehot(bond).to(device))
        
        if count == 0:
            return torch.zeros(self.bond_fdim, device=device, dtype=torch.float32)
        
        zero_tensor = torch.zeros(self.bond_fdim, device=device, dtype=torch.float32)
        for tensor in bondtype_list:
            zero_tensor += tensor
        return zero_tensor / count
    
    def get_atom_electronegativity(self, mol: Chem.Mol, atom_idx: int, 
                                   ep_list: Dict[str, float], 
                                   exclude_atom: int) -> float:
        """
        计算原子相对于其邻居的电负性差总和（参考 Retro-MTGR 的 get_Atom_EP）
        
        Parameters
        ----------
        mol: Chem.Mol
            RDKit 分子对象
        atom_idx: int
            原子索引
        ep_list: Dict[str, float]
            电负性字典
        exclude_atom: int
            排除的原子索引（通常是键的另一端）
        
        Returns
        -------
        float: 电负性差总和
        """
        atomsnum = mol.GetNumAtoms()
        ep = 0.0
        
        atom = mol.GetAtomWithIdx(atom_idx)
        atom_symbol = atom.GetSymbol()
        if atom_symbol not in ep_list:
            atom_symbol = 'unknown'
        ep_atom = ep_list.get(atom_symbol, 0.0)
        
        for i in range(atomsnum):
            if i == atom_idx or i == exclude_atom:
                continue
            bond = mol.GetBondBetweenAtoms(i, atom_idx)
            if bond is None:
                continue
            
            neighbor_atom = mol.GetAtomWithIdx(i)
            neighbor_symbol = neighbor_atom.GetSymbol()
            if neighbor_symbol not in ep_list:
                neighbor_symbol = 'unknown'
            ep_neighbor = ep_list.get(neighbor_symbol, 0.0)
            
            ep += ep_atom - ep_neighbor
        
        return ep
    
    def get_bond_energy(self, mol: Chem.Mol, atom_i: int, atom_j: int,
                       bond_energy_table: Optional[Dict[str, float]] = None) -> float:
        """
        获取键能（参考 Retro-MTGR 的 GetBondenergy）
        
        如果提供了 bond_energy_table，则从表中查找；否则返回 0.0
        
        Parameters
        ----------
        mol: Chem.Mol
            RDKit 分子对象
        atom_i: int
            原子 i 的索引
        atom_j: int
            原子 j 的索引
        bond_energy_table: Optional[Dict[str, float]]
            键能表，键名为 "原子1符号+键类型符号+原子2符号"，如 "C-O", "C=C"
        
        Returns
        -------
        float: 键能值，如果未提供表则返回 0.0
        """
        if bond_energy_table is None:
            return 0.0
        
        atom_i_symbol = mol.GetAtomWithIdx(atom_i).GetSymbol()
        atom_j_symbol = mol.GetAtomWithIdx(atom_j).GetSymbol()
        bond = mol.GetBondBetweenAtoms(atom_i, atom_j)
        
        if bond is None:
            return 0.0
        
        bond_type = bond.GetBondType()
        if str(bond_type) == 'SINGLE':
            bond_symbol = '-'
        elif str(bond_type) == 'DOUBLE':
            bond_symbol = '='
        elif str(bond_type) == 'TRIPLE':
            bond_symbol = '#'
        elif str(bond_type) == 'AROMATIC':
            bond_symbol = '~'
        else:
            return 0.0
        
        bond_name1 = atom_i_symbol + bond_symbol + atom_j_symbol
        bond_name2 = atom_j_symbol + bond_symbol + atom_i_symbol
        
        if bond_name1 in bond_energy_table:
            return float(bond_energy_table[bond_name1])
        elif bond_name2 in bond_energy_table:
            return float(bond_energy_table[bond_name2])
        else:
            return 0.0
    
    def forward(self, atom_features: torch.Tensor, 
                graph_tensors: Tuple[torch.Tensor],
                mol: Chem.Mol = None,
                ep_list: Optional[Dict[str, float]] = None,
                bond_energy_table: Optional[Dict[str, float]] = None) -> torch.Tensor:
        """
        前向传播：预测每条键的断裂概率
        
        支持 Retro-MTGR 风格的完整键特征构建
        
        Parameters
        ----------
        atom_features: torch.Tensor
            编码器输出的原子特征 [num_atoms, hidden_size]
        graph_tensors: Tuple[torch.Tensor]
            图张量元组，包含键信息
        mol: Chem.Mol, optional
            RDKit 分子对象，用于获取键类型信息
        ep_list: Optional[Dict[str, float]]
            电负性字典，用于计算电负性差特征
        bond_energy_table: Optional[Dict[str, float]]
            键能表，用于计算键能特征
        
        Returns
        -------
        bond_scores: torch.Tensor
            每条键的断裂概率 [num_bonds, 1]
        """
        f_atoms, f_bonds, a2b, b2a, b2revb, undirected_b2a = graph_tensors
        device = atom_features.device
        num_bonds = f_bonds.size(0)
        
        # 获取每条键的两个端原子索引
        atom_i_indices = undirected_b2a[:, 0]
        atom_j_indices = undirected_b2a[:, 1]
        
        # 获取端原子特征
        atom_i_feats = atom_features[atom_i_indices]  # [num_bonds, hidden_size]
        atom_j_feats = atom_features[atom_j_indices]  # [num_bonds, hidden_size]
        
        # 构建键特征（参考 Retro-MTGR 的 Redout_P_Bond_Test）
        bond_features_list = []
        
        # 1. 键类型特征（始终使用）
        if mol is not None:
            bond_type_feats = []
            for bond_idx in range(num_bonds):
                if bond_idx < mol.GetNumBonds():
                    bond = mol.GetBondWithIdx(bond_idx)
                    bond_type_onehot = self.get_bond_type_onehot(bond).to(device)
                else:
                    bond_type_onehot = torch.zeros(self.bond_fdim, device=device)
                bond_type_feats.append(bond_type_onehot)
            bond_type_feats = torch.stack(bond_type_feats)  # [num_bonds, bond_fdim]
        else:
            bond_type_feats = torch.zeros(num_bonds, self.bond_fdim, device=device)
        bond_type_feats = self.bond_type_linear(bond_type_feats)
        bond_features_list.append(bond_type_feats)
        
        # 2. 原子特征（根据是否使用原子符号选择拼接方式）
        if self.use_atom_symbol and mol is not None:
            # Retro-MTGR 方式：(atom_symbol + GCN_feat) + (atom_symbol + GCN_feat)
            atom_pair_feats = []
            for bond_idx in range(num_bonds):
                atom_i_idx = atom_i_indices[bond_idx].item()
                atom_j_idx = atom_j_indices[bond_idx].item()
                
                atom_i_symbol = mol.GetAtomWithIdx(atom_i_idx).GetSymbol()
                atom_j_symbol = mol.GetAtomWithIdx(atom_j_idx).GetSymbol()
                
                atom_i_symbol_feat = self.get_atom_symbol_onehot(atom_i_symbol, device)
                atom_j_symbol_feat = self.get_atom_symbol_onehot(atom_j_symbol, device)
                
                # (atom_symbol + GCN_feat) + (atom_symbol + GCN_feat)
                atom_i_combined = torch.cat([atom_i_symbol_feat, atom_i_feats[bond_idx]], dim=0)
                atom_j_combined = torch.cat([atom_j_symbol_feat, atom_j_feats[bond_idx]], dim=0)
                atom_pair_feat = atom_i_combined + atom_j_combined
                atom_pair_feats.append(atom_pair_feat)
            atom_pair_feats = torch.stack(atom_pair_feats)  # [num_bonds, atom_symbol_dim + hidden_size]
        else:
            # 简化方式：GCN 特征相加
            atom_pair_feats = atom_i_feats + atom_j_feats  # [num_bonds, hidden_size]
        bond_features_list.append(atom_pair_feats)
        
        # 3. 邻居键类型特征（可选）
        if self.use_neighbor_bond_type and mol is not None:
            neighbor_bond_type_feats = []
            for bond_idx in range(num_bonds):
                atom_i_idx = atom_i_indices[bond_idx].item()
                atom_j_idx = atom_j_indices[bond_idx].item()
                
                neighbor_i = self.get_neighbor_bond_type(mol, atom_i_idx, device)
                neighbor_j = self.get_neighbor_bond_type(mol, atom_j_idx, device)
                neighbor_feat = neighbor_i + neighbor_j  # 相加，参考 Retro-MTGR
                neighbor_bond_type_feats.append(neighbor_feat)
            neighbor_bond_type_feats = torch.stack(neighbor_bond_type_feats)  # [num_bonds, bond_fdim]
            neighbor_bond_type_feats = self.neighbor_bond_type_linear(neighbor_bond_type_feats)
            bond_features_list.append(neighbor_bond_type_feats)
        
        # 4. 电负性差特征（可选）
        if self.use_electronegativity and mol is not None and ep_list is not None:
            ep_feats = []
            for bond_idx in range(num_bonds):
                atom_i_idx = atom_i_indices[bond_idx].item()
                atom_j_idx = atom_j_indices[bond_idx].item()
                
                ep_i = self.get_atom_electronegativity(mol, atom_i_idx, ep_list, atom_j_idx)
                ep_j = self.get_atom_electronegativity(mol, atom_j_idx, ep_list, atom_i_idx)
                ep_diff = abs(ep_i - ep_j)  # 绝对值，参考 Retro-MTGR
                ep_feats.append(torch.tensor([ep_diff], device=device, dtype=torch.float32))
            ep_feats = torch.stack(ep_feats)  # [num_bonds, 1]
            ep_feats = self.electronegativity_linear(ep_feats)
            bond_features_list.append(ep_feats)
        
        # 5. 键能特征（可选）
        if self.use_bond_energy and mol is not None:
            # 计算所有键的键能，用于归一化
            if bond_energy_table is not None:
                bond_energies = []
                for bond_idx in range(num_bonds):
                    atom_i_idx = atom_i_indices[bond_idx].item()
                    atom_j_idx = atom_j_indices[bond_idx].item()
                    be = self.get_bond_energy(mol, atom_i_idx, atom_j_idx, bond_energy_table)
                    bond_energies.append(be)
                
                if len(bond_energies) > 0 and max(bond_energies) > 0:
                    max_be = max(bond_energies)
                    bond_energy_feats = torch.tensor(
                        [[be / max_be] for be in bond_energies], 
                        device=device, dtype=torch.float32
                    )  # [num_bonds, 1]
                else:
                    bond_energy_feats = torch.zeros(num_bonds, 1, device=device)
            else:
                bond_energy_feats = torch.zeros(num_bonds, 1, device=device)
            
            bond_energy_feats = self.bond_energy_linear(bond_energy_feats)
            bond_features_list.append(bond_energy_feats)
        
        # 拼接所有特征
        bond_features = torch.cat(bond_features_list, dim=1)  # [num_bonds, feature_dim]
        
        # 投影（如果特征维度太大）
        if self.projection is not None:
            bond_features = self.projection(bond_features)
        
        # 预测断裂概率
        bond_scores = self.predictor(bond_features)  # [num_bonds, 1]
        bond_probs = torch.sigmoid(bond_scores)  # [num_bonds, 1]
        
        return bond_probs


class ReactionCenterPredictor(nn.Module):
    """
    反应中心预测器：包装 ReactionCenterHead，处理批量数据
    """
    
    def __init__(self, hidden_size: int, bond_fdim: int = 4,
                 use_bond_energy: bool = False, dropout: float = 0.15):
        super(ReactionCenterPredictor, self).__init__()
        self.head = ReactionCenterHead(
            hidden_size=hidden_size,
            bond_fdim=bond_fdim,
            use_bond_energy=use_bond_energy,
            dropout=dropout
        )
    
    def forward(self, atom_features_list: List[torch.Tensor],
                graph_tensors_list: List[Tuple[torch.Tensor]],
                mols: List[Chem.Mol] = None) -> List[torch.Tensor]:
        """
        批量预测反应中心
        
        Parameters
        ----------
        atom_features_list: List[torch.Tensor]
            每个分子的原子特征列表
        graph_tensors_list: List[Tuple[torch.Tensor]]
            每个分子的图张量列表
        mols: List[Chem.Mol], optional
            RDKit 分子对象列表
        
        Returns
        -------
        bond_probs_list: List[torch.Tensor]
            每个分子的键断裂概率列表
        """
        bond_probs_list = []
        
        if mols is None:
            mols = [None] * len(atom_features_list)
        
        for atom_feats, graph_tensors, mol in zip(atom_features_list, graph_tensors_list, mols):
            bond_probs = self.head(atom_feats, graph_tensors, mol)
            bond_probs_list.append(bond_probs)
        
        return bond_probs_list

