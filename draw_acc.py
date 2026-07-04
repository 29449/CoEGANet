import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import os
from matplotlib import rcParams


def load_data(excel_path):
    """加载Excel数据，提取epoch和模型训练/验证数据"""
    xls = pd.ExcelFile(excel_path)
    data = {}

    for sheet_name in xls.sheet_names:
        df = pd.read_excel(excel_path, sheet_name=sheet_name)

        # 检测epoch列（优先匹配含'epoch'的列）
        epoch_col = None
        for col in df.columns:
            if 'epoch' in col.lower():
                epoch_col = col
                break
        if not epoch_col:
            print(f"警告：表单 '{sheet_name}' 中未找到epoch列，使用第一列 '{df.columns[0]}'")
            epoch_col = df.columns[0]

        sheet_data = {'epoch': df[epoch_col].values}

        # 提取模型数据（从表头中解析模型名称）
        for col in df.columns:
            if col == epoch_col:
                continue  # 跳过epoch列

            # 从表头中提取模型名称（新格式：train w/o α (base) -> w/o α (base)）
            if col.startswith('train '):
                model_name = col[len('train '):].strip()
                data_type = 'train'
            elif col.startswith('valid '):
                model_name = col[len('valid '):].strip()
                data_type = 'valid'
            else:
                continue  # 只处理train/valid列

            # 初始化模型数据存储
            if model_name not in sheet_data:
                sheet_data[model_name] = {}
            sheet_data[model_name][data_type] = df[col].values

        data[sheet_name] = sheet_data

    return data


def plot_combined_accuracy(data, sheet_name, output_dir):
    """绘制训练和验证准确率曲线，确保颜色与模型匹配"""
    plt.figure(figsize=(12, 8))

    # 设置学术图表样式
    rcParams.update({
        'font.family': 'Arial',
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.labelweight': 'bold',
        'axes.titlesize': 16,
        'axes.titleweight': 'bold',
        'legend.fontsize': 10,
        'figure.dpi': 300,
        'axes.grid': True,
        'grid.alpha': 0.3
    })

    epochs = data['epoch']

    # 颜色配置：根据新的模型名称
    colors = {
        'w/o α (base)': '#ff7f0e',        # 橙色 - 基准模型
        'w/ α (h0-α)': '#1f77b4',         # 蓝色 - h0-α
        'w/ α (hl-α)': '#2ca02c',         # 绿色 - hl-α
        # 'w/ α (h0|hl-α)': '#d62728',      # 红色 - h0|hl-α
        # 'w/ α (SG(h0|hl)-α)': '#9467bd',  # 紫色 - SG(h0|hl)-α
        'w/ α (VG(h0|hl)-α)': '#8c564b',  # 棕色 - VG(h0|hl)-α
    }

    # 线型配置
    line_styles = {
        'train': '--',  # 训练用虚线
        'valid': '-'    # 验证用实线
    }

    # 打印当前表单的模型名称（用于调试匹配情况）
    model_names = [k for k in data.keys() if k != 'epoch']
    print(f"表单 '{sheet_name}' 的模型名称：{model_names}")

    # 绘制所有模型的训练和验证曲线
    for model in model_names:
        color = colors.get(model, '#888888')
        
        # 训练曲线（虚线）
        if 'train' in data[model]:
            plt.plot(epochs, data[model]['train'],
                     color=color,
                     linestyle='--',
                     linewidth=1.5,
                     label=f'{model} (Train)',
                     alpha=0.8)

        # 验证曲线（实线）
        if 'valid' in data[model]:
            plt.plot(epochs, data[model]['valid'],
                     color=color,
                     linestyle='-',
                     linewidth=2,
                     label=f'{model} (Valid)',
                     alpha=0.8)

    # 设置图表标题和坐标轴
    reaction_type = 'Without' if 'without' in sheet_name.lower() else 'With'
    plt.title(f'Model Performance Comparison ({reaction_type} Reaction)', fontsize=16, pad=18)
    plt.xlabel('Training Epochs', fontsize=14, labelpad=10, fontweight='bold')
    plt.ylabel('Accuracy', fontsize=14, labelpad=10, fontweight='bold')

    # 坐标轴范围和刻度
    plt.xlim(0, 150)
    plt.xticks([0, 25, 50, 75, 100, 125, 150])
    
    # 根据数据范围调整y轴
    if 'without' in sheet_name.lower():
        plt.ylim(0.25, 0.85)  # without sheet的数据范围
    else:
        plt.ylim(0.35, 0.95)  # with sheet的数据范围

    # 网格和边框设置
    plt.grid(True, linestyle='-', color='#999999', alpha=0.3)
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_color('#666666')
        spine.set_linewidth(1.5)

    # 图例设置：两列，右下角
    plt.legend(loc='lower right', frameon=True,
               facecolor='white', edgecolor='gray',
               framealpha=1.0, borderpad=0.8,
               ncol=2, fontsize=9)

    # 保存图表
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'α_{sheet_name}.pdf')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved {output_path}')


def main():
    excel_path = "attn.xlsx"  # 替换为你的Excel文件路径
    output_dir = "results"

    # 加载数据
    data_dict = load_data(excel_path)

    # 为每个表单绘制图表
    for sheet_name, sheet_data in data_dict.items():
        plot_combined_accuracy(sheet_data, sheet_name, output_dir)


if __name__ == "__main__":
    main()
