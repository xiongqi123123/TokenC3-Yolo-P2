"""
正确的TokenC3-LSNet稀疏化模型验证脚本
直接加载稀疏化模型，避免权重匹配问题
"""

import torch
import torch.nn as nn
import time
import os
import argparse
import json
from PIL import Image
from ultralytics import YOLO
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def count_nonzero_parameters(model):
    """统计非零参数数量"""
    total_params = 0
    nonzero_params = 0
    
    for param in model.parameters():
        total_params += param.numel()
        nonzero_params += (param != 0).sum().item()
    
    return total_params, nonzero_params


def analyze_sparsity(model):
    """分析模型稀疏度"""
    print(f"\n📊 稀疏度分析:")
    
    total_params = 0
    total_zeros = 0
    layer_count = 0
    
    for name, param in model.named_parameters():
        if 'weight' in name:
            params = param.numel()
            zeros = (param == 0).sum().item()
            sparsity = zeros / params * 100
            
            total_params += params
            total_zeros += zeros
            layer_count += 1
            
            if sparsity > 1.0:  # 只显示稀疏度超过1%的层
                print(f"  {name}: {sparsity:.2f}% 稀疏度 ({zeros:,}/{params:,})")
    
    overall_sparsity = total_zeros / total_params * 100 if total_params > 0 else 0
    print(f"  整体稀疏度: {overall_sparsity:.2f}% ({total_zeros:,}/{total_params:,})")
    print(f"  稀疏化层数: {layer_count}")
    
    return overall_sparsity


def load_sparse_model(model_path: str, pth_path: str, device: str = 'cuda:4'):
    """加载稀疏化模型"""
    print(f"\n🔧 加载稀疏化模型...")
    print(f"📁 原始模型: {model_path}")
    print(f"📁 稀疏化权重: {pth_path}")
    
    try:
        # 加载原始模型结构
        model = YOLO(model_path)
        model.to(device)
        model.eval()
        
        # 加载稀疏化权重
        sparse_state_dict = torch.load(pth_path, map_location=device)
        model.model.load_state_dict(sparse_state_dict, strict=True)
        
        print(f"✅ 稀疏化模型加载成功")
        return model
        
    except Exception as e:
        print(f"❌ 稀疏化模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def validate_model_inference(model, device: str = 'cuda:4', num_runs: int = 100):
    """验证模型推理性能"""
    print(f"\n⚡ 推理性能测试...")
    
    model.eval()
    
    # 创建测试输入
    test_input = torch.randn(1, 3, 640, 640).to(device)
    
    # 预热
    print(f"🔥 模型预热 (10次)...")
    for _ in range(10):
        with torch.no_grad():
            outputs = model(test_input)
    
    # 性能测试
    print(f"📊 推理速度测试 ({num_runs}次)...")
    torch.cuda.synchronize() if device.startswith('cuda') else None
    start_time = time.time()
    
    for _ in range(num_runs):
        with torch.no_grad():
            outputs = model(test_input)
    
    torch.cuda.synchronize() if device.startswith('cuda') else None
    end_time = time.time()
    
    # 计算性能指标
    total_time = end_time - start_time
    avg_time = total_time / num_runs * 1000  # 转换为毫秒
    fps = 1000 / avg_time
    
    print(f"📊 推理性能:")
    print(f"  ⏱️ 平均推理时间: {avg_time:.2f} ms")
    print(f"  🎯 FPS: {fps:.1f}")
    print(f"  📦 输入形状: {test_input.shape}")
    
    # 检查输出
    if isinstance(outputs, (list, tuple)):
        print(f"  📊 输出数量: {len(outputs)}")
        for i, output in enumerate(outputs):
            if hasattr(output, 'shape'):
                print(f"  📊 输出{i}形状: {output.shape}")
    else:
        print(f"  📊 输出形状: {outputs.shape if hasattr(outputs, 'shape') else type(outputs)}")
    
    return avg_time, fps


def yolo_to_coco(yolo_dir, img_dir, class_names, output_json):
    """YOLO标签转COCO格式"""
    images = []
    annotations = []
    ann_id = 1
    img_id = 1
    
    for img_name in sorted(os.listdir(img_dir)):
        if not img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
            continue
        img_path = os.path.join(img_dir, img_name)
        label_path = os.path.join(yolo_dir, os.path.splitext(img_name)[0] + '.txt')
        if not os.path.exists(label_path):
            continue
            
        img = Image.open(img_path)
        width, height = img.size
        images.append({
            "file_name": img_name,
            "height": height,
            "width": width,
            "id": img_id
        })
        
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls, x, y, w, h = map(float, parts)
                x1 = (x - w / 2) * width
                y1 = (y - h / 2) * height
                w_box = w * width
                h_box = h * height
                annotations.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": int(cls) + 1,
                    "bbox": [x1, y1, w_box, h_box],
                    "area": w_box * h_box,
                    "iscrowd": 0
                })
                ann_id += 1
        img_id += 1
        
    categories = [{"id": i+1, "name": name} for i, name in enumerate(class_names)]
    coco_dict = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }
    
    with open(output_json, 'w') as f:
        json.dump(coco_dict, f, indent=4)
    
    return coco_dict


def validate_model_accuracy_with_coco(model, data_yaml: str, device: str = 'cuda:4'):
    """使用pycocotools验证模型检测精度"""
    print(f"\n🎯 检测精度验证 (使用pycocotools)...")
    
    # 解析dataset.yaml获取测试集路径
    import yaml
    with open(data_yaml, 'r') as f:
        data_config = yaml.safe_load(f)
    
    dataset_root = data_config['path']
    test_images_dir = os.path.join(dataset_root, data_config['val'])  # val实际指向test
    test_labels_dir = test_images_dir.replace('images', 'labels')
    class_names = data_config['names']
    
    if not os.path.exists(test_images_dir):
        print(f"❌ 测试集图片目录不存在: {test_images_dir}")
        return None
    
    if not os.path.exists(test_labels_dir):
        print(f"❌ 测试集标签目录不存在: {test_labels_dir}")
        return None
    
    try:
        # 1. 转换标签为COCO格式
        print(f"📝 转换YOLO标签为COCO格式...")
        coco_ann_path = 'temp_annotations_coco.json'
        coco_res_path = 'temp_results_coco.json'
        
        yolo_to_coco(test_labels_dir, test_images_dir, class_names, coco_ann_path)
        
        # 2. 推理并保存为COCO格式预测
        print(f"🔮 模型推理...")
        results = []
        img_id = 1
        
        for img_name in sorted(os.listdir(test_images_dir)):
            if not img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
                continue
            img_path = os.path.join(test_images_dir, img_name)
            
            # 使用模型推理
            pred = model(img_path)[0]
            boxes = pred.boxes.cpu().numpy()
            
            for box, score, cls in zip(boxes.xyxy, boxes.conf, boxes.cls):
                x1, y1, x2, y2 = box
                w = x2 - x1
                h = y2 - y1
                results.append({
                    "image_id": img_id,
                    "category_id": int(cls) + 1,
                    "bbox": [float(x1), float(y1), float(w), float(h)],
                    "score": float(score)
                })
            img_id += 1
        
        with open(coco_res_path, 'w') as f:
            json.dump(results, f)
        
        # 3. 用pycocotools评估
        print(f"📊 使用pycocotools评估...")
        cocoGt = COCO(coco_ann_path)
        cocoDt = cocoGt.loadRes(coco_res_path)
        cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        
        # 提取关键指标
        metrics = {
            'mAP50-95': float(cocoEval.stats[0]),  # AP @0.5:0.95
            'mAP50': float(cocoEval.stats[1]),     # AP @0.5
            'precision': 0.0,  # COCO评估不直接提供precision
            'recall': 0.0      # COCO评估不直接提供recall
        }
        
        print(f"📊 检测精度 (pycocotools):")
        print(f"  🎯 mAP50-95: {metrics['mAP50-95']:.4f}")
        print(f"  🎯 mAP50: {metrics['mAP50']:.4f}")
        
        # 清理临时文件
        if os.path.exists(coco_ann_path):
            os.remove(coco_ann_path)
        if os.path.exists(coco_res_path):
            os.remove(coco_res_path)
        
        return metrics
        
    except Exception as e:
        print(f"❌ 精度验证失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def compare_with_original(original_path: str, sparse_path: str, data_path: str, device: str = 'cuda:4'):
    """与原始模型对比"""
    print(f"\n📊 与原始模型对比...")
    
    try:
        # 加载原始模型
        print(f"🔧 加载原始模型...")
        original_model = YOLO(original_path)
        original_model.to(device)
        
        # 统计原始模型参数
        orig_total, orig_nonzero = count_nonzero_parameters(original_model.model)
        
        # 加载稀疏化模型
        sparse_model = load_sparse_model(original_path, sparse_path, device)
        if sparse_model is None:
            return
        
        # 统计稀疏化模型参数
        sparse_total, sparse_nonzero = count_nonzero_parameters(sparse_model.model)
        
        # 计算减少比例
        param_reduction = (orig_total - sparse_total) / orig_total * 100
        effective_reduction = (orig_nonzero - sparse_nonzero) / orig_nonzero * 100
        
        print(f"📊 参数对比:")
        print(f"  🔵 原始模型总参数: {orig_total:,}")
        print(f"  🔵 原始模型非零参数: {orig_nonzero:,}")
        print(f"  🟢 稀疏化模型总参数: {sparse_total:,}")
        print(f"  🟢 稀疏化模型非零参数: {sparse_nonzero:,}")
        print(f"  📉 总参数减少: {param_reduction:.2f}%")
        print(f"  📉 有效参数减少: {effective_reduction:.2f}%")
        
        # 性能对比
        print(f"\n⚡ 推理速度对比...")
        orig_time, orig_fps = validate_model_inference(original_model, device, 50)
        sparse_time, sparse_fps = validate_model_inference(sparse_model, device, 50)
        
        speed_improvement = (orig_time - sparse_time) / orig_time * 100
        fps_improvement = (sparse_fps - orig_fps) / orig_fps * 100
        
        print(f"📊 速度对比:")
        print(f"  🔵 原始模型: {orig_time:.2f} ms, {orig_fps:.1f} FPS")
        print(f"  🟢 稀疏化模型: {sparse_time:.2f} ms, {sparse_fps:.1f} FPS")
        print(f"  🚀 速度提升: {speed_improvement:.2f}%, FPS提升: {fps_improvement:.2f}%")
        
        # 精度对比
        if os.path.exists(data_path):
            print(f"\n🎯 检测精度对比...")
            orig_metrics = validate_model_accuracy_with_coco(original_model, data_path, device)
            sparse_metrics = validate_model_accuracy_with_coco(sparse_model, data_path, device)
            
            if orig_metrics and sparse_metrics:
                print(f"\n📊 精度对比:")
                for metric in ['mAP50-95', 'mAP50']:
                    orig_val = orig_metrics[metric]
                    sparse_val = sparse_metrics[metric]
                    change = ((sparse_val - orig_val) / orig_val * 100) if orig_val > 0 else 0
                    status = "✅" if change > -2 else "⚠️" if change > -5 else "❌"
                    print(f"  {status} {metric}: {orig_val:.4f} → {sparse_val:.4f} ({change:+.2f}%)")
        
        return {
            'param_reduction': param_reduction,
            'effective_reduction': effective_reduction,
            'speed_improvement': speed_improvement,
            'fps_improvement': fps_improvement
        }
        
    except Exception as e:
        print(f"❌ 对比分析失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description='正确验证TokenC3-LSNet稀疏化模型')
    parser.add_argument('--sparse_model', type=str, 
                       default='tokenc3_pure_sparsity_10p.pth',
                       help='稀疏化模型权重文件路径')
    parser.add_argument('--original_model', type=str,
                       default='/home/qi.xiong/Test/yolov13_tokenc3_lsnet_p2/yolov13_UAV_Sheep/tokenc3_lsnet_p2/weights/best.pt',
                       help='原始模型路径')
    parser.add_argument('--data', type=str,
                       default='/home/qi.xiong/Dataset/UAV_Sheep/yolo_dataset/dataset.yaml',
                       help='数据集配置文件路径')
    parser.add_argument('--device', type=str, default='cuda:4',
                       help='推理设备')
    parser.add_argument('--compare', action='store_true',
                       help='是否与原始模型对比')
    
    args = parser.parse_args()
    
    print("🚀 TokenC3-LSNet稀疏化模型正确验证")
    print("="*60)
    
    # 1. 加载稀疏化模型
    sparse_model = load_sparse_model(args.original_model, args.sparse_model, args.device)
    if sparse_model is None:
        print("❌ 稀疏化模型加载失败，退出验证")
        return
    
    # 2. 分析稀疏度
    sparsity = analyze_sparsity(sparse_model.model)
    
    # 3. 验证推理性能
    avg_time, fps = validate_model_inference(sparse_model, args.device)
    
    # 4. 验证检测精度
    metrics = validate_model_accuracy_with_coco(sparse_model, args.data, args.device)
    
    # 5. 统计参数信息
    total_params, nonzero_params = count_nonzero_parameters(sparse_model.model)
    effective_reduction = (1 - nonzero_params / total_params) * 100
    
    # 6. 与原始模型对比（可选）
    comparison_results = None
    if args.compare:
        comparison_results = compare_with_original(
            args.original_model, args.sparse_model, args.data, args.device
        )
    
    # 7. 总结结果
    print(f"\n🎉 稀疏化模型验证完成!")
    print("="*60)
    print(f"📊 核心结果:")
    print(f"  🕸️ 整体稀疏度: {sparsity:.2f}%")
    print(f"  📉 有效参数减少: {effective_reduction:.2f}%")
    print(f"  ⚡ 推理时间: {avg_time:.2f} ms")
    print(f"  🎯 FPS: {fps:.1f}")
    
    if metrics:
        print(f"  🎯 mAP50: {metrics['mAP50']:.4f}")
        print(f"  🎯 mAP50-95: {metrics['mAP50-95']:.4f}")
    
    if comparison_results:
        print(f"  📈 相比原始模型:")
        print(f"    有效参数减少: {comparison_results['effective_reduction']:.2f}%")
        print(f"    速度提升: {comparison_results['speed_improvement']:.2f}%")
    
    # 评估结果
    success = True
    issues = []
    
    if sparsity < 8.0:
        issues.append("稀疏度不足8%")
        success = False
    
    if metrics and metrics['mAP50'] < 0.8:
        issues.append("mAP50低于0.8")
        success = False
    
    if avg_time > 25:
        issues.append("推理时间超过25ms")
        success = False
    
    if success:
        print(f"\n✅ 验证成功! 稀疏化模型满足所有要求。")
    else:
        print(f"\n⚠️ 验证发现问题: {', '.join(issues)}")
        print(f"建议调整稀疏化策略或参数。")


if __name__ == "__main__":
    main() 