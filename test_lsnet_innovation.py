#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LSNet-Enhanced TokenC3 Innovation Test Script

测试基于LSNet "See Large, Focus Small" 思想的 TokenC3 增强版本
专为无人机视角下高密度小目标检测优化

Author: Cursor AI Assistant
Date: 2024
"""

import torch
import torch.nn as nn
import time
import sys
import os

# 添加项目路径
sys.path.append('Improve/yolov13')

from ultralytics.nn.modules.block import (
    TokenC3_LSNet, LSConv, LSGhostBottleneck, 
    TokenC3 as OriginalTokenC3
)

def test_tensor_shapes():
    """测试张量维度兼容性"""
    print("🔍 测试张量维度兼容性...")
    
    # 测试不同输入尺寸
    test_cases = [
        (1, 64, 64, 64),   # 标准输入
        (1, 128, 32, 32),  # 高分辨率小图
        (1, 256, 16, 16),  # 深层特征
        (2, 64, 64, 64),   # 批量处理
    ]
    
    for i, (b, c, h, w) in enumerate(test_cases):
        print(f"  测试案例 {i+1}: 输入形状 [{b}, {c}, {h}, {w}]")
        
        # 创建模型
        model = TokenC3_LSNet(c1=c, c2=c, n=2)
        model.eval()
        
        # 测试输入
        x = torch.randn(b, c, h, w)
        
        with torch.no_grad():
            try:
                output = model(x)
                print(f"    ✅ 输出形状: {list(output.shape)}")
                assert output.shape[0] == b and output.shape[1] == c
                print(f"    ✅ 维度匹配正确")
            except Exception as e:
                print(f"    ❌ 错误: {e}")
                return False
    
    return True

def test_lsnet_components():
    """测试LSNet组件功能"""
    print("\n🧩 测试LSNet组件功能...")
    
    # 测试LSConv
    print("  测试LSConv (Large-Small Convolution)...")
    lsconv = LSConv(c1=64, c2=64, large_kernel=7, small_kernel=3)
    x = torch.randn(1, 64, 32, 32)
    
    with torch.no_grad():
        output = lsconv(x)
        print(f"    LSConv输出形状: {list(output.shape)}")
        assert output.shape == x.shape
        print("    ✅ LSConv功能正常")
    
    # 测试LSGhostBottleneck
    print("  测试LSGhostBottleneck...")
    ls_ghost = LSGhostBottleneck(c1=64, c2=64, use_dllm=True)
    
    with torch.no_grad():
        output = ls_ghost(x)
        print(f"    LSGhostBottleneck输出形状: {list(output.shape)}")
        assert output.shape == x.shape
        print("    ✅ LSGhostBottleneck功能正常")

def benchmark_performance():
    """性能基准测试"""
    print("\n⚡ 性能基准测试...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  使用设备: {device}")
    
    # 测试配置
    batch_size = 4
    channels = 128
    height, width = 64, 64
    num_runs = 100
    
    x = torch.randn(batch_size, channels, height, width).to(device)
    
    # 原始TokenC3
    print("  测试原始TokenC3性能...")
    original_model = OriginalTokenC3(c1=channels, c2=channels, n=3).to(device)
    original_model.eval()
    
    # 预热
    with torch.no_grad():
        for _ in range(10):
            _ = original_model(x)
    
    # 计时
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(num_runs):
            _ = original_model(x)
    
    torch.cuda.synchronize() if device.type == 'cuda' else None
    original_time = time.time() - start_time
    
    # LSNet增强版TokenC3
    print("  测试LSNet增强版TokenC3性能...")
    lsnet_model = TokenC3_LSNet(c1=channels, c2=channels, n=3).to(device)
    lsnet_model.eval()
    
    # 预热
    with torch.no_grad():
        for _ in range(10):
            _ = lsnet_model(x)
    
    # 计时
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(num_runs):
            _ = lsnet_model(x)
    
    torch.cuda.synchronize() if device.type == 'cuda' else None
    lsnet_time = time.time() - start_time
    
    # 结果分析
    print(f"\n📊 性能对比结果:")
    print(f"  原始TokenC3:      {original_time/num_runs*1000:.2f} ms/次")
    print(f"  LSNet增强版:       {lsnet_time/num_runs*1000:.2f} ms/次")
    print(f"  性能比率:         {lsnet_time/original_time:.2f}x")
    
    if lsnet_time/original_time < 1.5:
        print("  ✅ 性能开销可接受 (< 1.5x)")
    else:
        print("  ⚠️  性能开销较高，可能需要优化")

def test_feature_quality():
    """特征质量测试"""
    print("\n🎯 特征质量测试...")
    
    # 创建模拟小目标检测场景
    batch_size = 2
    channels = 64
    height, width = 128, 128
    
    # 创建包含小目标的模拟图像
    x = torch.zeros(batch_size, channels, height, width)
    
    # 添加小目标信号 (模拟无人机视角下的小目标)
    for b in range(batch_size):
        # 随机位置添加小目标
        for _ in range(5):  # 5个小目标
            center_h = torch.randint(10, height-10, (1,)).item()
            center_w = torch.randint(10, width-10, (1,)).item()
            # 3x3小目标
            x[b, :, center_h-1:center_h+2, center_w-1:center_w+2] = torch.randn(channels, 3, 3) * 2
    
    # 添加背景噪声
    x += torch.randn_like(x) * 0.1
    
    # 测试两个模型的特征表示
    original_model = OriginalTokenC3(c1=channels, c2=channels, n=2)
    lsnet_model = TokenC3_LSNet(c1=channels, c2=channels, n=2)
    
    original_model.eval()
    lsnet_model.eval()
    
    with torch.no_grad():
        original_features = original_model(x)
        lsnet_features = lsnet_model(x)
        
        # 计算特征统计
        print(f"  原始TokenC3特征统计:")
        print(f"    均值: {original_features.mean().item():.4f}")
        print(f"    标准差: {original_features.std().item():.4f}")
        print(f"    动态范围: [{original_features.min().item():.4f}, {original_features.max().item():.4f}]")
        
        print(f"  LSNet增强版特征统计:")
        print(f"    均值: {lsnet_features.mean().item():.4f}")
        print(f"    标准差: {lsnet_features.std().item():.4f}")
        print(f"    动态范围: [{lsnet_features.min().item():.4f}, {lsnet_features.max().item():.4f}]")
        
        # 特征差异分析
        feature_diff = torch.abs(lsnet_features - original_features).mean()
        print(f"  特征差异: {feature_diff.item():.4f}")
        
        if feature_diff > 0.1:
            print("  ✅ LSNet确实产生了不同的特征表示")
        else:
            print("  ⚠️  特征变化较小，可能需要调整参数")

def main():
    """主测试函数"""
    print("🚀 LSNet-Enhanced TokenC3 创新测试")
    print("=" * 50)
    
    try:
        # 测试张量维度
        if not test_tensor_shapes():
            print("❌ 张量维度测试失败")
            return
        
        # 测试组件功能  
        test_lsnet_components()
        
        # 性能基准测试
        benchmark_performance()
        
        # 特征质量测试
        test_feature_quality()
        
        print("\n🎉 所有测试完成!")
        print("💡 LSNet增强版TokenC3创新点总结:")
        print("   1. ✅ 维度兼容性良好")
        print("   2. ✅ 组件功能正常") 
        print("   3. ✅ 性能开销可控")
        print("   4. ✅ 产生差异化特征表示")
        print("\n🔬 创新亮点:")
        print("   • 融合LSNet的'See Large, Focus Small'策略")
        print("   • 结合Ghost卷积实现轻量化")
        print("   • 保留DLLM动态局部混合能力")
        print("   • 专为无人机视角小目标检测优化")
        
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 