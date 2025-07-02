#!/usr/bin/env python3
"""
TokenC3-LSNet智能剪枝运行脚本
参考原版YOLOv13的成功策略，针对TokenC3-LSNet优化
"""

import os
import sys
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='TokenC3-LSNet智能剪枝')
    parser.add_argument('--model', type=str, 
                       default="/home/qi.xiong/Test/yolov13_tokenc3_lsnet_p2/yolov13_UAV_Sheep/tokenc3_lsnet_p2/weights/best.pt",
                       help='模型权重路径')
    parser.add_argument('--target_sparsity', type=float, default=0.1,
                       help='目标稀疏度 (0.1 = 10%参数减少)')
    parser.add_argument('--device', type=str, default='cuda:4',
                       help='运行设备')
    parser.add_argument('--output', type=str, default='tokenc3_lsnet_pruned_10p.pth',
                       help='输出文件名')
    
    args = parser.parse_args()
    
    print("🚀 启动TokenC3-LSNet智能剪枝系统")
    print(f"📁 模型路径: {args.model}")
    print(f"🎯 目标稀疏度: {args.target_sparsity*100:.1f}%")
    print(f"💻 运行设备: {args.device}")
    print(f"💾 输出文件: {args.output}")
    print("="*60)
    
    # 导入并运行剪枝
    try:
        from pure_sparsity_pruning import PureSparsityPruner
        
        # 初始化剪枝器
        pruner = PureSparsityPruner(
            model_path=args.model,
            target_sparsity=args.target_sparsity,
            device=args.device
        )
        
        # 执行剪枝
        results = pruner.pure_sparsity_prune()
        
        # 分析分布
        pruner.analyze_sparsity_distribution()
        
        # 验证结构
        if pruner.validate_structure():
            # 保存模型
            saved_path = pruner.save_pruned_model(args.output)
            
            if saved_path and results['effective_reduction'] > 0.08:
                print(f"\n🎉 TokenC3-LSNet剪枝成功完成!")
                print(f"📊 参数减少: {results['effective_reduction']*100:.2f}%")
                print(f"💾 保存位置: {saved_path}")
                print(f"\n📝 下一步验证:")
                print(f"python correct_validation.py \\")
                print(f"  --model_path {saved_path} \\")
                print(f"  --data_path /home/qi.xiong/Dataset/UAV_Sheep/yolo_dataset/dataset.yaml")
                return True
            else:
                print(f"\n⚠️ 剪枝效果不达标，请调整参数")
                return False
        else:
            print(f"\n❌ 模型结构验证失败")
            return False
            
    except Exception as e:
        print(f"\n❌ 剪枝过程出错: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 