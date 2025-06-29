#!/usr/bin/env python3
"""快速特征差异测试"""

import torch
import sys
sys.path.append('Improve/yolov13')

from ultralytics.nn.modules.block import TokenC3_LSNet, TokenC3 as OriginalTokenC3

def test_feature_difference():
    print("🔬 专项特征差异测试")
    
    # 创建更极端的测试场景
    batch_size = 1
    channels = 128
    height, width = 64, 64
    
    # 创建具有明显空间模式的输入
    x = torch.zeros(batch_size, channels, height, width)
    
    # 添加大尺度模式 (适合大核捕获)
    for i in range(0, height, 8):
        for j in range(0, width, 8):
            x[:, :, i:i+4, j:j+4] = torch.randn(channels, 4, 4) * 3
    
    # 添加小尺度细节 (适合小核捕获)  
    for i in range(channels):
        x[:, i, torch.randint(0, height, (10,)), torch.randint(0, width, (10,))] = 5.0
    
    # 测试两个模型
    original_model = OriginalTokenC3(c1=channels, c2=channels, n=2)
    lsnet_model = TokenC3_LSNet(c1=channels, c2=channels, n=2)
    
    original_model.eval()
    lsnet_model.eval()
    
    with torch.no_grad():
        original_features = original_model(x)
        lsnet_features = lsnet_model(x)
        
        # 详细特征分析
        print(f"📊 详细特征分析:")
        
        # 基础统计
        print(f"  原始TokenC3:")
        print(f"    均值: {original_features.mean().item():.6f}")
        print(f"    标准差: {original_features.std().item():.6f}")
        print(f"    最大值: {original_features.max().item():.6f}")
        print(f"    最小值: {original_features.min().item():.6f}")
        
        print(f"  LSNet增强版:")
        print(f"    均值: {lsnet_features.mean().item():.6f}")
        print(f"    标准差: {lsnet_features.std().item():.6f}")
        print(f"    最大值: {lsnet_features.max().item():.6f}")
        print(f"    最小值: {lsnet_features.min().item():.6f}")
        
        # 特征差异指标
        abs_diff = torch.abs(lsnet_features - original_features)
        rel_diff = abs_diff / (torch.abs(original_features) + 1e-8)
        
        print(f"  🎯 差异分析:")
        print(f"    平均绝对差异: {abs_diff.mean().item():.6f}")
        print(f"    最大绝对差异: {abs_diff.max().item():.6f}")
        print(f"    平均相对差异: {rel_diff.mean().item():.6f}")
        print(f"    差异标准差: {abs_diff.std().item():.6f}")
        
        # 激活模式分析
        orig_active = (original_features > original_features.mean()).float().mean()
        lsnet_active = (lsnet_features > lsnet_features.mean()).float().mean()
        
        print(f"  🔥 激活模式:")
        print(f"    原始激活率: {orig_active.item():.4f}")
        print(f"    LSNet激活率: {lsnet_active.item():.4f}")
        print(f"    激活模式差异: {abs(orig_active - lsnet_active).item():.4f}")
        
        # 评估创新效果
        if abs_diff.mean().item() > 0.05:
            print("  ✅ 显著特征差异 - 创新效果明显!")
        elif abs_diff.mean().item() > 0.02:
            print("  ⚡ 中等特征差异 - 创新有效果")
        else:
            print("  ⚠️  特征差异较小 - 需要进一步调整")

if __name__ == "__main__":
    test_feature_difference() 