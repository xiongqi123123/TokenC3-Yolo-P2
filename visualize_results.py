#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Experiment Results Visualization Tool
Author: AI Assistant
Date: 2024
Description: 支持多个实验结果的对比可视化
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib import rcParams
import seaborn as sns
from math import pi
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# 设置图形样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

class ResultsVisualizer:
    """实验结果可视化工具类"""
    
    def __init__(self, output_dir: str = '/home/qi.xiong/Improve/yolov13/logs'):
        """
        初始化可视化工具
        
        Args:
            output_dir: 输出目录路径
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 定义指标配置
        self.metrics_config = {
            # 精度指标 (越高越好)
            'metrics/precision(B)': {'name': 'Precision', 'higher_better': True, 'color': '#1f77b4'},
            'metrics/recall(B)': {'name': 'Recall', 'higher_better': True, 'color': '#ff7f0e'},
            'metrics/mAP50(B)': {'name': 'mAP@0.5', 'higher_better': True, 'color': '#2ca02c'},
            'metrics/mAP75(B)': {'name': 'mAP@0.75', 'higher_better': True, 'color': '#d62728'},
            'metrics/mAP50-95(B)': {'name': 'mAP@0.5:0.95', 'higher_better': True, 'color': '#9467bd'},
            
            # 损失指标 (越低越好)
            'train/box_loss': {'name': 'Train Box Loss', 'higher_better': False, 'color': '#8c564b'},
            'train/cls_loss': {'name': 'Train Cls Loss', 'higher_better': False, 'color': '#e377c2'},
            'train/dfl_loss': {'name': 'Train DFL Loss', 'higher_better': False, 'color': '#7f7f7f'},
            'val/box_loss': {'name': 'Val Box Loss', 'higher_better': False, 'color': '#bcbd22'},
            'val/cls_loss': {'name': 'Val Cls Loss', 'higher_better': False, 'color': '#17becf'},
            'val/dfl_loss': {'name': 'Val DFL Loss', 'higher_better': False, 'color': '#ff9896'}
        }
        
        # 定义颜色调色板
        self.color_palette = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
        ]
        
        # 定义线型样式
        self.line_styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]
    
    def load_experiment_data(self, experiment_configs: Dict[str, str]) -> Dict[str, pd.DataFrame]:
        """
        加载多个实验的数据
        
        Args:
            experiment_configs: 实验配置字典 {'experiment_name': 'csv_file_path'}
            
        Returns:
            实验数据字典 {'experiment_name': DataFrame}
        """
        experiments_data = {}
        min_epochs = float('inf')
        
        for exp_name, csv_path in experiment_configs.items():
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                experiments_data[exp_name] = df
                min_epochs = min(min_epochs, len(df))
                print(f"✅ 加载 {exp_name}: {len(df)} epochs from {csv_path}")
            else:
                print(f"❌ 文件不存在: {csv_path}")
        
        # 确保所有实验的长度一致
        for exp_name in experiments_data:
            experiments_data[exp_name] = experiments_data[exp_name].iloc[:min_epochs]
        
        print(f"📊 统一训练轮数为: {min_epochs} epochs")
        return experiments_data
    
    def calculate_improvement(self, val1: float, val2: float, higher_better: bool = True) -> float:
        """计算改进百分比"""
        if val2 == 0:
            return 0.0
        
        improvement = ((val1 - val2) / val2) * 100
        
        if not higher_better:  # 对于loss类指标，越低越好
            improvement = -improvement
        
        return improvement
    
    def plot_metrics_comparison(self, experiments_data: Dict[str, pd.DataFrame], 
                              metrics: List[str], 
                              title: str,
                              filename: str,
                              baseline_name: Optional[str] = None) -> None:
        """绘制多个实验的指标对比图"""
        n_metrics = len(metrics)
        n_cols = 3
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        plt.figure(figsize=(20, 6 * n_rows))
        
        for i, metric in enumerate(metrics, 1):
            plt.subplot(n_rows, n_cols, i)
            
            config = self.metrics_config.get(metric, {'name': metric, 'higher_better': True})
            
            # 绘制每个实验的曲线
            for j, (exp_name, df) in enumerate(experiments_data.items()):
                if metric in df.columns:
                    color = self.color_palette[j % len(self.color_palette)]
                    linestyle = self.line_styles[j % len(self.line_styles)]
                    
                    plt.plot(df['epoch'], df[metric], 
                           label=exp_name, linewidth=2.5, 
                           color=color, alpha=0.8, linestyle=linestyle)
            
            plt.title(f'{config["name"]} Comparison', fontsize=14, fontweight='bold')
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel(config["name"], fontsize=12)
            plt.legend(fontsize=10)
            plt.grid(True, alpha=0.3)
            
            # 如果指定了baseline，计算并显示改进百分比
            if baseline_name and baseline_name in experiments_data:
                baseline_df = experiments_data[baseline_name]
                if metric in baseline_df.columns:
                    baseline_val = baseline_df[metric].iloc[-1]
                    
                    # 显示其他实验相对于baseline的改进
                    y_pos = 0.9
                    for exp_name, df in experiments_data.items():
                        if exp_name != baseline_name and metric in df.columns:
                            exp_val = df[metric].iloc[-1]
                            improvement = self.calculate_improvement(
                                exp_val, baseline_val, config.get('higher_better', True)
                            )
                            
                            color = 'green' if improvement > 0 else 'red'
                            symbol = '↑' if improvement > 0 else '↓'
                            
                            plt.text(0.02, y_pos, f'{exp_name}: {symbol} {improvement:+.2f}%', 
                                   transform=plt.gca().transAxes,
                                   fontsize=9, color=color, fontweight='bold',
                                   bbox=dict(boxstyle="round,pad=0.2", 
                                           facecolor='lightgreen' if improvement > 0 else 'lightcoral', 
                                           alpha=0.7))
                            y_pos -= 0.1
        
        plt.suptitle(title, fontsize=18, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/{filename}', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📊 已保存: {filename}")
    
    def plot_accuracy_metrics(self, experiments_data: Dict[str, pd.DataFrame], 
                            baseline_name: Optional[str] = None) -> None:
        """绘制精度指标对比"""
        accuracy_metrics = ['metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)', 
                           'metrics/mAP75(B)', 'metrics/mAP50-95(B)']
        
        self.plot_metrics_comparison(
            experiments_data, accuracy_metrics,
            'Multi-Experiment Accuracy Metrics Comparison',
            'accuracy_metrics_comparison.png',
            baseline_name
        )
    
    def plot_loss_metrics(self, experiments_data: Dict[str, pd.DataFrame], 
                         baseline_name: Optional[str] = None) -> None:
        """绘制损失指标对比"""
        loss_metrics = ['train/box_loss', 'train/cls_loss', 'train/dfl_loss', 
                       'val/box_loss', 'val/cls_loss', 'val/dfl_loss']
        
        self.plot_metrics_comparison(
            experiments_data, loss_metrics,
            'Multi-Experiment Loss Metrics Comparison',
            'loss_metrics_comparison.png',
            baseline_name
        )
    
    def plot_compact_comparison(self, experiments_data: Dict[str, pd.DataFrame], 
                              baseline_name: Optional[str] = None) -> None:
        """
        生成紧凑型训练对比图 - 解决多实验标注重叠问题
        """
        # 定义要显示的指标（按你图片中的顺序）
        metrics_layout = [
            # 第一行
            ['metrics/mAP50(B)', 'metrics/mAP75(B)', 'metrics/mAP50-95(B)', 'metrics/precision(B)'],
            # 第二行  
            ['metrics/recall(B)', 'train/box_loss', 'train/cls_loss', 'train/dfl_loss']
        ]
        
        # 创建图形
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle('Multi-Experiment Training Comparison', fontsize=18, fontweight='bold', y=0.98)
        
        # 定义固定的baseline颜色
        baseline_color = '#808080'  # 灰色，代表基准
        baseline_linestyle = '-'    # 实线样式
        
        # 获取非baseline实验列表，用于按颜色表分配颜色
        non_baseline_experiments = [name for name in experiments_data.keys() if name != baseline_name]
        
        # 为非baseline实验分配颜色
        experiment_colors = {}
        experiment_linestyles = {}
        
        # 首先设置baseline
        if baseline_name and baseline_name in experiments_data:
            experiment_colors[baseline_name] = baseline_color
            experiment_linestyles[baseline_name] = baseline_linestyle
        
        # 然后为其他实验分配颜色 - 都使用虚线
        for i, exp_name in enumerate(non_baseline_experiments):
            experiment_colors[exp_name] = self.color_palette[i % len(self.color_palette)]
            experiment_linestyles[exp_name] = '--'  # 所有非baseline实验都使用虚线
        
        for row_idx, row_metrics in enumerate(metrics_layout):
            for col_idx, metric in enumerate(row_metrics):
                ax = axes[row_idx, col_idx]
                
                # 获取指标配置
                config = self.metrics_config.get(metric, {'name': metric, 'higher_better': True})
                
                # 绘制每个实验的曲线
                for exp_name, df in experiments_data.items():
                    if metric in df.columns:
                        color = experiment_colors.get(exp_name, '#1f77b4')
                        linestyle = experiment_linestyles.get(exp_name, '-')
                        linewidth = 2.5 if exp_name != baseline_name else 2.0
                        alpha = 0.9 if exp_name != baseline_name else 0.7
                        
                        ax.plot(df['epoch'], df[metric], 
                               label=exp_name, linewidth=linewidth, 
                               color=color, alpha=alpha, linestyle=linestyle)
                
                # 设置标题和标签
                ax.set_title(config['name'], fontsize=12, fontweight='bold', pad=10)
                ax.set_xlabel('Epoch', fontsize=10)
                ax.set_ylabel(config['name'], fontsize=10)
                
                # 设置网格
                ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
                ax.set_facecolor('#f8f9fa')
                
                # 添加图例（只在第一个子图添加）
                if row_idx == 0 and col_idx == 0:
                    ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
                
                # 计算并显示改进百分比 - 解决重叠问题
                if baseline_name and baseline_name in experiments_data:
                    baseline_df = experiments_data[baseline_name]
                    if metric in baseline_df.columns:
                        baseline_val = baseline_df[metric].iloc[-1]
                        
                        # 收集所有非baseline实验的改进信息
                        improvements = []
                        for exp_name, df in experiments_data.items():
                            if exp_name != baseline_name and metric in df.columns:
                                exp_val = df[metric].iloc[-1]
                                improvement = self.calculate_improvement(
                                    exp_val, baseline_val, config.get('higher_better', True)
                                )
                                improvements.append({
                                    'name': exp_name,
                                    'improvement': improvement,
                                    'value': exp_val
                                })
                        
                        # 根据实验数量决定显示策略
                        if len(improvements) == 1:
                            # 单个实验：右上角显示
                            imp = improvements[0]
                            color = '#28a745' if imp['improvement'] > 0 else '#dc3545'
                            symbol = '↑' if imp['improvement'] > 0 else '↓'
                            improvement_text = f"{symbol} {imp['improvement']:+.1f}%"
                            
                            bbox_props = dict(
                                boxstyle="round,pad=0.25", 
                                facecolor='lightgreen' if imp['improvement'] > 0 else 'lightcoral', 
                                alpha=0.8, edgecolor=color, linewidth=1
                            )
                            
                            ax.text(0.98, 0.95, improvement_text,
                                   transform=ax.transAxes, fontsize=10, color=color, 
                                   fontweight='bold', ha='right', va='top', bbox=bbox_props)
                        
                        elif len(improvements) == 2:
                            # 两个实验：右上角和右中显示
                            positions = [(0.98, 0.95), (0.98, 0.80)]
                            for i, imp in enumerate(improvements):
                                color = '#28a745' if imp['improvement'] > 0 else '#dc3545'
                                symbol = '↑' if imp['improvement'] > 0 else '↓'
                                improvement_text = f"{imp['name']}: {symbol} {imp['improvement']:+.1f}%"
                                
                                bbox_props = dict(
                                    boxstyle="round,pad=0.2", 
                                    facecolor='lightgreen' if imp['improvement'] > 0 else 'lightcoral', 
                                    alpha=0.8, edgecolor=color, linewidth=1
                                )
                                
                                ax.text(positions[i][0], positions[i][1], improvement_text,
                                       transform=ax.transAxes, fontsize=9, color=color, 
                                       fontweight='bold', ha='right', va='top', bbox=bbox_props)
                        
                        else:
                            # 多个实验：垂直排列在右侧
                            start_y = 0.95
                            for i, imp in enumerate(improvements):
                                y_pos = start_y - i * 0.12
                                if y_pos < 0.2:  # 如果太低，跳过
                                    break
                                    
                                color = '#28a745' if imp['improvement'] > 0 else '#dc3545'
                                symbol = '↑' if imp['improvement'] > 0 else '↓'
                                
                                # 简化显示：只显示实验名和百分比
                                short_name = imp['name'].replace('TokenC3_', 'TC3_')  # 缩短名称
                                improvement_text = f"{short_name}: {symbol}{imp['improvement']:+.1f}%"
                                
                                bbox_props = dict(
                                    boxstyle="round,pad=0.15", 
                                    facecolor='lightgreen' if imp['improvement'] > 0 else 'lightcoral', 
                                    alpha=0.8, edgecolor=color, linewidth=0.8
                                )
                                
                                ax.text(0.98, y_pos, improvement_text,
                                       transform=ax.transAxes, fontsize=8, color=color, 
                                       fontweight='bold', ha='right', va='top', bbox=bbox_props)
                        
                        # 在左下角显示baseline的最终值
                        baseline_text = f'{baseline_name}: {baseline_val:.4f}'
                        ax.text(0.02, 0.05, baseline_text,
                               transform=ax.transAxes, fontsize=9, 
                               color=experiment_colors.get(baseline_name, baseline_color),
                               ha='left', va='bottom',
                               bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
                
                # 美化坐标轴
                ax.tick_params(axis='both', which='major', labelsize=9)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_color('#cccccc')
                ax.spines['bottom'].set_color('#cccccc')
        
        # 调整布局
        plt.tight_layout()
        plt.subplots_adjust(top=0.93, hspace=0.3, wspace=0.3)
        
        # 保存图片
        plt.savefig(f'{self.output_dir}/compact_training_comparison.png', 
                    dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        print(f"📊 已保存紧凑型对比图: compact_training_comparison.png")
    
    def visualize_all(self, experiments_data: Dict[str, pd.DataFrame], 
                     baseline_name: Optional[str] = None) -> None:
        """一键生成所有可视化图表"""
        print("🚀 开始生成多实验对比可视化...")
        
        # 1. 生成紧凑型对比图 (你要的样式)
        self.plot_compact_comparison(experiments_data, baseline_name)
        
        # 2. 精度指标对比
        self.plot_accuracy_metrics(experiments_data, baseline_name)
        
        # 3. 损失指标对比
        self.plot_loss_metrics(experiments_data, baseline_name)
        
        print(f"📊 所有图表已保存到: {self.output_dir}")
        print("✅ 可视化完成！")


# 便捷函数
def compare_multiple_experiments(experiment_configs: Dict[str, str],
                               baseline_name: Optional[str] = None,
                               output_dir: str = '/home/qi.xiong/Improve/yolov13/logs') -> None:
    """
    多实验对比的便捷函数
    
    Args:
        experiment_configs: 实验配置字典 {'实验名': 'CSV路径'}
        baseline_name: 基准实验名称
        output_dir: 输出目录
    """
    visualizer = ResultsVisualizer(output_dir)
    data = visualizer.load_experiment_data(experiment_configs)
    visualizer.visualize_all(data, baseline_name)


# 使用示例
if __name__ == "__main__":

    # 创建可视化工具
    visualizer = ResultsVisualizer()

    # 定义实验
    experiments = {
        'Yolov8n': "/home/qi.xiong/Improve/yolov8/yolov8_UAV_Sheep/baseline_yolov8n/results.csv",
        # 'Yolo11n': "/home/qi.xiong/Improve/yolo11/yolov11_UAV_Sheep/baseline_yolov11n/results.csv",
        # 'Yolov10n': "/home/qi.xiong/Improve/yolov10/yolov10_UAV_Sheep/baseline_yolov10n/results.csv",
        # 'Yolo12n': "/home/qi.xiong/Improve/yolov12/yolo12_UAV_Sheep/baseline_yolo12n/results.csv",
        # 'TokenC3_NoMosaic': "/home/qi.xiong/Improve/yolov13/yolov13_UAV_Sheep/tokenc3_NoMosaic/results.csv",
        'TokenC3_LSNet': "/home/qi.xiong/Improve/yolov13/yolov13_UAV_Sheep/tokenc3_lsnet/results.csv",
        'TokenC3_LSNet_P2': "/home/qi.xiong/Improve/yolov13/yolov13_UAV_Sheep/tokenc3_lsnet_p2/results.csv",
        'Baseline_NoMosaic': '/home/qi.xiong/Improve/yolov13/yolov13_UAV_Sheep/baseline_NoMosaic/results.csv',
    }

    # 加载数据并生成紧凑型对比图
    data = visualizer.load_experiment_data(experiments)
    visualizer.plot_compact_comparison(data, baseline_name='Baseline_NoMosaic') 