"""
TokenC3-LSNet纯权重稀疏化系统
只进行权重稀疏化，保持模型结构完整性，实现10%参数减少
"""

import torch
import torch.nn as nn
import numpy as np
import copy
from ultralytics import YOLO
import os
from pathlib import Path


class PureSparsityPruner:
    """TokenC3-LSNet智能剪枝器（结合权重稀疏化和模块感知策略）"""
    
    def __init__(self, model_path: str, target_sparsity: float = 0.1, device: str = 'cuda:4'):
        self.model_path = model_path
        self.target_sparsity = target_sparsity
        self.device = device
        
        print(f"🧠 初始化TokenC3纯权重稀疏化器")
        print(f"📁 模型路径: {model_path}")
        print(f"🎯 目标稀疏度: {target_sparsity*100:.1f}%")
        print(f"💻 设备: {self.device}")
        print(f"🔧 策略: 纯权重稀疏化（保持结构完整）")
        
        # 加载模型
        self.model = self._load_model()
        self.original_params = self._count_parameters()
        self.original_nonzero = self._count_nonzero_parameters()
        
        # 分析模型结构
        self._analyze_model_structure()
        
        print(f"✅ 模型加载成功")
        print(f"📊 原始参数量: {self.original_params:,}")
        print(f"📊 原始非零参数: {self.original_nonzero:,}")
        
    def _load_model(self):
        """加载模型到指定设备"""
        try:
            # 参考原版yolov13的成功做法
            yolo_wrapper = YOLO(self.model_path)
            yolo_wrapper.to(self.device)
            model = yolo_wrapper.model
            model.eval()
            
            # 检查模型是否加载成功
            if model is None:
                raise RuntimeError("模型加载失败：model为None")
            
            print(f"✅ 模型成功加载到设备: {self.device}")
            return model
        except Exception as e:
            raise RuntimeError(f"模型加载失败: {e}")
    
    def _count_parameters(self):
        """统计模型参数数量"""
        return sum(p.numel() for p in self.model.parameters())
    
    def _count_nonzero_parameters(self):
        """统计非零参数数量"""
        return sum((p != 0).sum().item() for p in self.model.parameters())
    
    def _analyze_model_structure(self):
        """分析模型结构"""
        self.sparsity_layers = []
        
        print(f"\n🔍 分析模型结构...")
        
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                # 排除检测头（保持检测精度）
                if not self._is_detection_head(name):
                    self.sparsity_layers.append((name, module))
        
        print(f"📊 可稀疏化层数: {len(self.sparsity_layers)}")
        
        # 按模块类型统计
        self._categorize_layers()
        
    def _categorize_layers(self):
        """按模块类型分类层"""
        self.layer_categories = {
            'tokenc3': [],
            'hyperace': [],
            'fullpad': [],
            'dllm': [],
            'dsc3': [],
            'a2c2f': [],
            'backbone': [],
            'neck': [],
            'other': []
        }
        
        for name, module in self.sparsity_layers:
            name_lower = name.lower()
            
            if any(pattern in name_lower for pattern in ['tokenc3', 'lsconv', 'lsghost']):
                self.layer_categories['tokenc3'].append((name, module))
            elif 'hyperace' in name_lower:
                self.layer_categories['hyperace'].append((name, module))
            elif any(pattern in name_lower for pattern in ['fullpad', 'tunnel']):
                self.layer_categories['fullpad'].append((name, module))
            elif 'dllm' in name_lower:
                self.layer_categories['dllm'].append((name, module))
            elif any(pattern in name_lower for pattern in ['dsc3']):
                self.layer_categories['dsc3'].append((name, module))
            elif 'a2c2f' in name_lower:
                self.layer_categories['a2c2f'].append((name, module))
            elif any(pattern in name_lower for pattern in ['model.0', 'model.1', 'model.2', 'model.3', 'model.4', 'model.5', 'model.6', 'model.7', 'model.8']):
                self.layer_categories['backbone'].append((name, module))
            elif any(pattern in name_lower for pattern in ['model.1', 'model.2', 'model.3']):
                self.layer_categories['neck'].append((name, module))
            else:
                self.layer_categories['other'].append((name, module))
        
        # 打印分类统计
        for category, layers in self.layer_categories.items():
            if len(layers) > 0:
                print(f"📊 {category.upper()}模块: {len(layers)} 层")
    
    def _is_detection_head(self, layer_name: str) -> bool:
        """判断是否为检测头"""
        return 'model.31' in layer_name
    
    def _get_sparsity_ratio(self, layer_name: str) -> float:
        """
        获取层的稀疏化比例
        参考原版YOLOv13剪枝的成功策略，针对TokenC3-LSNet特殊模块优化
        """
        name_lower = layer_name.lower()
        
        # 🛡️ 超重要模块 - 极保守剪枝
        if any(pattern in name_lower for pattern in ['tokenc3', 'lsconv', 'lsghost']):
            return 0.06  # TokenC3创新模块：6%稀疏化（稍微提高）
        elif 'hyperace' in name_lower:
            return 0.04  # HyperACE：4%稀疏化（最重要）
        elif any(pattern in name_lower for pattern in ['lsnet', 'ls_']):
            return 0.08  # LSNet创新部分：8%稀疏化
        
        # 🔧 重要模块 - 保守剪枝  
        elif any(pattern in name_lower for pattern in ['fullpad', 'tunnel']):
            return 0.10  # FullPAD：10%稀疏化
        elif any(pattern in name_lower for pattern in ['a2c2f', 'attn']):
            return 0.12  # 注意力模块：12%稀疏化
        elif 'dllm' in name_lower:
            return 0.12  # DLLM：12%稀疏化（提高一些）
        
        # 📊 标准模块 - 适度剪枝
        elif any(pattern in name_lower for pattern in ['dsc3', 'dsc2']):
            return 0.15  # DSC模块：15%稀疏化
        elif any(pattern in name_lower for pattern in ['model.0', 'model.1', 'model.2']):
            return 0.06  # 骨干早期层：6%稀疏化（重要）
        elif any(pattern in name_lower for pattern in ['model.22', 'model.24']):
            return 0.18  # 上采样层：18%稀疏化
        
        # ⚡ 可以大胆剪枝的层
        elif 'conv' in name_lower and not any(x in name_lower for x in ['tokenc3', 'lsconv']):
            return 0.22  # 普通卷积：22%稀疏化（更积极）
        else:
            return 0.12  # 其他层：12%稀疏化
    
    def pure_sparsity_prune(self):
        """执行纯权重稀疏化"""
        print(f"\n🔬 开始纯权重稀疏化...")
        
        nonzero_before = self._count_nonzero_parameters()
        success_count = 0
        total_weights_processed = 0
        total_weights_sparsified = 0
        
        # 按类别处理层
        for category, layers in self.layer_categories.items():
            if len(layers) == 0:
                continue
                
            print(f"\n🔧 处理{category.upper()}模块 ({len(layers)}层)...")
            category_success = 0
            
            for name, module in layers:
                try:
                    if hasattr(module, 'weight') and module.weight is not None:
                        weight = module.weight.data
                        layer_sparsity = self._get_sparsity_ratio(name)
                        
                        num_weights = weight.numel()
                        num_to_sparse = int(num_weights * layer_sparsity)
                        
                        if num_to_sparse > 0:
                            # 计算权重重要性（使用L1 norm）
                            importance = torch.abs(weight)
                            flat_importance = importance.view(-1)
                            _, indices_to_sparse = torch.topk(flat_importance, num_to_sparse, largest=False)
                            
                            # 稀疏化权重
                            flat_weight = weight.view(-1)
                            flat_weight[indices_to_sparse] = 0.0
                            module.weight.data = flat_weight.view(weight.shape)
                            
                            total_weights_processed += num_weights
                            total_weights_sparsified += num_to_sparse
                            category_success += 1
                            success_count += 1
                    
                except Exception as e:
                    print(f"    ❌ {name}: 稀疏化失败 - {e}")
            
            if category_success > 0:
                category_sparsity = total_weights_sparsified / total_weights_processed if total_weights_processed > 0 else 0
                print(f"    ✅ {category.upper()}: {category_success}/{len(layers)} 层成功")
        
        nonzero_after = self._count_nonzero_parameters()
        effective_reduction = (nonzero_before - nonzero_after) / self.original_nonzero
        actual_sparsity = total_weights_sparsified / total_weights_processed if total_weights_processed > 0 else 0
        
        results = {
            'original_params': self.original_params,
            'final_params': self._count_parameters(),
            'original_nonzero': self.original_nonzero,
            'final_nonzero': nonzero_after,
            'weights_sparsified': total_weights_sparsified,
            'total_weights': total_weights_processed,
            'actual_sparsity': actual_sparsity,
            'effective_reduction': effective_reduction,
            'target_sparsity': self.target_sparsity,
            'layers_processed': success_count,
            'total_layers': len(self.sparsity_layers)
        }
        
        self._print_results(results)
        return results
    
    def _print_results(self, results):
        """打印结果统计"""
        print(f"\n🎯 纯权重稀疏化完成!")
        print(f"📊 结果统计:")
        print(f"  🔵 原始参数: {results['original_params']:,}")
        print(f"  🟢 最终参数: {results['final_params']:,}")
        print(f"  📊 原始非零参数: {results['original_nonzero']:,}")
        print(f"  📊 最终非零参数: {results['final_nonzero']:,}")
        print(f"  🕸️ 稀疏化权重数: {results['weights_sparsified']:,}")
        print(f"  📉 实际稀疏度: {results['actual_sparsity']*100:.2f}%")
        print(f"  📉 有效参数减少: {results['effective_reduction']*100:.2f}% (目标: {results['target_sparsity']*100:.1f}%)")
        print(f"  ✅ 成功处理层数: {results['layers_processed']}/{results['total_layers']}")
        
        # 额外统计信息
        params_reduced = results['original_params'] - results['final_params']
        model_size_reduction = params_reduced / results['original_params']
        print(f"  📦 模型大小减少: {model_size_reduction*100:.2f}%")
    
    def validate_structure(self):
        """验证稀疏化后模型结构"""
        try:
            test_input = torch.randn(1, 3, 640, 640).to(self.device)
            
            print(f"🔍 模型结构验证...")
            
            with torch.no_grad():
                output = self.model(test_input)
            
            print(f"✅ 模型结构验证通过")
            print(f"📊 输入形状: {test_input.shape}")
            if isinstance(output, (list, tuple)):
                print(f"📊 输出数量: {len(output)}")
                for i, o in enumerate(output):
                    if hasattr(o, 'shape'):
                        print(f"📊 输出{i}形状: {o.shape}")
            
            return True
            
        except Exception as e:
            print(f"❌ 模型结构验证失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def analyze_sparsity_distribution(self):
        """分析稀疏化分布"""
        print(f"\n📊 稀疏化分布分析:")
        
        total_params = 0
        total_zeros = 0
        
        for category, layers in self.layer_categories.items():
            if len(layers) == 0:
                continue
                
            category_params = 0
            category_zeros = 0
            
            for name, module in layers:
                if hasattr(module, 'weight') and module.weight is not None:
                    weight = module.weight.data
                    params = weight.numel()
                    zeros = (weight == 0).sum().item()
                    
                    category_params += params
                    category_zeros += zeros
            
            if category_params > 0:
                category_sparsity = category_zeros / category_params
                print(f"  {category.upper()}: {category_sparsity*100:.2f}% 稀疏度 ({category_zeros:,}/{category_params:,})")
                
                total_params += category_params
                total_zeros += category_zeros
        
        if total_params > 0:
            overall_sparsity = total_zeros / total_params
            print(f"  整体稀疏度: {overall_sparsity*100:.2f}%")
    
    def save_pruned_model(self, save_path: str):
        """保存稀疏化后的模型"""
        try:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            
            torch.save(self.model.state_dict(), save_path)
            
            file_size_mb = os.path.getsize(save_path) / 1024 / 1024
            
            print(f"💾 稀疏化模型已保存")
            print(f"📁 路径: {save_path}")
            print(f"📦 大小: {file_size_mb:.1f} MB")
            
            return save_path
            
        except Exception as e:
            print(f"❌ 模型保存失败: {e}")
            return None


def main():
    """TokenC3-LSNet智能剪枝主函数"""
    model_path = "/home/qi.xiong/Test/yolov13_tokenc3_lsnet_p2/yolov13_UAV_Sheep/tokenc3_lsnet_p2/weights/best.pt"
    save_path = "tokenc3_lsnet_pruned_10p.pth"  # 更直观的命名
    device = "cuda:4"
    
    # 验证文件路径
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        print(f"请确认模型文件路径正确")
        return False
    
    print(f"📄 模型路径验证通过: {model_path}")
    
    try:
        # 初始化纯权重稀疏化器
        pruner = PureSparsityPruner(model_path, target_sparsity=0.1, device=device)
        
        # 执行纯权重稀疏化
        results = pruner.pure_sparsity_prune()
        
        # 分析稀疏化分布
        pruner.analyze_sparsity_distribution()
        
        # 验证模型结构
        if pruner.validate_structure():
            # 保存稀疏化模型
            saved_path = pruner.save_pruned_model(save_path)
            
            if saved_path and results['effective_reduction'] > 0.08:  # 至少减少8%有效参数
                print(f"\n🎉 TokenC3-LSNet智能剪枝成功!")
                print(f"📊 有效参数减少: {results['effective_reduction']*100:.2f}%")
                print(f"💾 模型保存: {save_path}")
                print(f"\n📝 下一步验证精度:")
                print(f"  python correct_validation.py")
                print(f"    --model_path {save_path}")
                print(f"    --data_path /home/qi.xiong/Dataset/UAV_Sheep/yolo_dataset/dataset.yaml")
                return True
            else:
                print(f"\n⚠️ 稀疏化效果不理想，需要调整策略")
        
        return False
        
    except Exception as e:
        print(f"❌ 纯权重稀疏化失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    if success:
        print(f"\n✅ 纯权重稀疏化实验成功!")
    else:
        print(f"\n❌ 纯权重稀疏化实验失败!") 