import numpy as np
import argparse
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from collections import defaultdict
import os
from matplotlib.colors import Normalize

# 官方 USPTO-50k 类别
rxn_class_names = [
    "Heteroatom alkylation and arylation",
    "Acylation and related processes",
    "C-C bond formation",
    "Heterocycle formation",
    "Protections",
    "Deprotections",
    "Reductions",
    "Oxidations",
    "Functional group interconversion (FGI)",
    "Functional group addition (FGA)"
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--npy_file', type=str, default="rse_vectors_final.npy")
    parser.add_argument('--target_rxn_class', type=int, default=0,
                        help="反应类型编号（0-9）")
    args = parser.parse_args()

    rse_records = np.load(args.npy_file, allow_pickle=True)

    # 按反应 idx 分组
    records_by_idx = defaultdict(list)
    for rec in rse_records:
        if rec['rxn_class'] == args.target_rxn_class:
            records_by_idx[rec['idx']].append(rec)

    if len(records_by_idx) == 0:
        print("No reactions found.")
        return

    # ===== 关键步骤 1：统计该反应类型下的最大步数 =====
    T_max = max(len(recs) for recs in records_by_idx.values())

    # 统一的颜色归一化
    norm = Normalize(vmin=1, vmax=T_max)
    cmap = plt.get_cmap('viridis')

    plt.figure(figsize=(8, 6))

    # ===== 绘制散点 =====
    for recs in records_by_idx.values():
        vecs = np.array([r['rse'].flatten() for r in recs])

        if vecs.shape[1] > 2:
            vecs_2d = PCA(n_components=2).fit_transform(vecs)
        elif vecs.shape[1] == 2:
            vecs_2d = vecs
        else:
            vecs_2d = np.hstack([vecs, np.zeros((vecs.shape[0], 1))])

        t = np.arange(1, len(recs) + 1)

        plt.scatter(
            vecs_2d[:, 0],
            vecs_2d[:, 1],
            c=t,
            cmap=cmap,
            norm=norm,
            s=80,
            marker='o',
            alpha=0.9
        )

    # ===== 统一的 colorbar =====
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm)
    cbar.set_label("edit step (t)")
    cbar.set_ticks(np.arange(1, T_max + 1))

    plt.xlabel("RSE dim 1")
    plt.ylabel("RSE dim 2")
    plt.title(
        f"RSE scatter plot for reaction type:\n"
        f"{rxn_class_names[args.target_rxn_class - 1]}"
    )

    os.makedirs("results", exist_ok=True)
    out_path = f"results/rse_scatter_rxn{args.target_rxn_class}.pdf"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved {out_path}")

if __name__ == '__main__':
    main()
