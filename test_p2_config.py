#!/usr/bin/env python3
"""
测试P2配置文件的有效性
"""

import torch
import sys
import os
sys.path.append('/home/qi.xiong/Improve/yolov13')

def test_p2_config():
    """测试P2配置文件"""
    print("🔍 测试P2配置文件...")
    
    try:
        from ultralytics import YOLO
        
        # 测试配置文件加载
        config_path = "/home/qi.xiong/Improve/yolov13/ultralytics/cfg/models/v13/yolov13_lsnet_p2_simple.yaml"
        
        print(f"加载配置文件: {config_path}")
        
        # 创建模型（不加载预训练权重）
        model = YOLO(config_path)
        print("✅ 配置文件加载成功")
        
        # 测试模型创建
        print(f"模型类型: {type(model.model)}")
        
        # 测试前向传播
        print("🔍 测试前向传播...")
        x = torch.randn(1, 3, 640, 640)
        print(f"输入: {x.shape}")
        
        with torch.no_grad():
            output = model.model(x)
            
        if isinstance(output, (list, tuple)):
            print(f"✅ 输出类型: {type(output)}, 长度: {len(output)}")
            for i, out in enumerate(output):
                if hasattr(out, 'shape'):
                    print(f"  输出[{i}]: {out.shape}")
                else:
                    print(f"  输出[{i}]: {type(out)}")
        else:
            print(f"✅ 输出: {output.shape}")
            
        print("✅ P2配置测试成功！")
        return True
        
    except Exception as e:
        print(f"❌ P2配置测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_summary():
    """测试模型结构摘要"""
    print("\n🔍 测试模型结构...")
    
    try:
        from ultralytics import YOLO
        
        config_path = "/home/qi.xiong/Improve/yolov13/ultralytics/cfg/models/v13/yolov13_lsnet_p2_simple.yaml"
        model = YOLO(config_path)
        
        # 打印模型信息
        print("模型参数统计:")
        total_params = sum(p.numel() for p in model.model.parameters())
        trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
        print(f"  总参数: {total_params:,}")
        print(f"  可训练参数: {trainable_params:,}")
        
        return True
        
    except Exception as e:
        print(f"❌ 模型结构测试失败: {e}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("P2配置文件测试")
    print("=" * 50)
    
    success1 = test_p2_config()
    success2 = test_model_summary()
    
    print("\n" + "=" * 50)
    if success1 and success2:
        print("✅ 所有测试通过！P2配置可以使用")
    else:
        print("❌ 测试失败，需要修复配置")
    print("=" * 50) 