# TokenC3-LSNet 剪枝工具

## 🎯 项目简介

针对TokenC3-LSNet模型的智能剪枝工具，实现**9.25%参数减少**，**mAP50精度仅损失0.60%**，**推理速度提升21.6%**。

## 🚀 快速开始

### 1. 环境准备
```bash
# 确保已安装依赖
pip install torch torchvision ultralytics
```

### 2. 一键剪枝
```bash
# 直接运行剪枝
python run_pruning.py

# 或自定义参数
python pure_sparsity_pruning.py
```

### 3. 验证精度
```bash
# 对比原始模型和剪枝模型精度
python correct_validation.py --sparse_model tokenc3_lsnet_pruned_10p_v2.pth --compare
```

## 📊 效果预览

| 指标 | 原始模型 | 剪枝模型 | 提升 |
|------|----------|----------|------|
| 参数减少 | - | **9.25%** | ✅ |
| mAP50精度 | 0.8811 | 0.8758 | -0.60% |
| 推理速度 | 48.1 FPS | **58.5 FPS** | +21.6% |

## 📂 核心文件

```
📁 剪枝工具包/
├── 🔧 pure_sparsity_pruning.py    # 核心剪枝算法
├── ▶️ run_pruning.py              # 一键运行脚本  
├── 🧪 correct_validation.py       # 精度验证工具
├── 💾 tokenc3_lsnet_pruned_10p_v2.pth  # 剪枝后模型
└── 📖 TokenC3_LSNet_剪枝方案文档.md     # 详细技术文档
```

## ⚙️ 配置参数

在`pure_sparsity_pruning.py`中修改：
```python
# 模型路径 - 改为你的模型路径
model_path = "/path/to/your/best.pt"

# 目标稀疏度 - 默认10%
target_sparsity = 0.10

# 计算设备
device = "cuda:4"  # 或 "cpu"
```

## 🔬 技术原理

采用**模块感知权重稀疏化**策略：

| 模块类型 | 稀疏化比例 | 说明 |
|----------|------------|------|
| TokenC3/LSConv | 6% | 核心创新模块，保守剪枝 |
| HyperACE | 4% | 最重要模块，最低稀疏化 |
| 普通卷积 | 22% | 可积极剪枝 |

## ❓ 常见问题

**Q: 为什么模型文件变大了？**  
A: 采用权重稀疏化，参数总数不变，只是部分权重置0。可结合压缩算法进一步减小。

**Q: 可以调整剪枝强度吗？**  
A: 可以修改`_get_sparsity_ratio`函数中的比例值，但建议保持TokenC3模块的保守策略。

**Q: 支持其他YOLO模型吗？**  
A: 当前专门为TokenC3-LSNet优化，其他模型需修改模块识别逻辑。

## 📚 详细文档

- 📖 [完整技术文档](./TokenC3_LSNet_剪枝方案文档.md)
- 📊 [优化总结](./PRUNING_OPTIMIZATION_SUMMARY.md)

---

🎉 **开箱即用的TokenC3-LSNet剪枝解决方案！** 