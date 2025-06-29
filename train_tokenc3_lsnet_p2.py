from ultralytics import YOLO
import logging
import os
# 恢复训练：修改model_config、resume、exist_ok
# 设置日志
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/train_tokenc3_lsnet_p2.log'),
        logging.StreamHandler()  # 同时输出到控制台
    ]
)

# 创建模型 - 使用正确的路径
model_config = '/home/qi.xiong/Improve/yolov13/ultralytics/cfg/models/v13/yolov13n_lsnet_p2_simple.yaml'

# model_config = '/home/qi.xiong/Improve/yolov13/yolov13_small_objects/baseline/weights/last.pt'
model = YOLO(model_config)

logging.info(f"Using model config: {model_config}")
logging.info(f"Starting tokenc3_lsnet_p2 training...")

# 训练参数 - 针对小目标优化
results = model.train(
    # ====== 基础参数 ======
    data='/home/qi.xiong/Dataset/UAV_Sheep/yolo_dataset/dataset.yaml',  # 数据集配置文件路径
    epochs=500,                # 训练轮数
    batch=64,                 # 批次大小
    imgsz=640,                 # 输入图片尺寸
    device="4,5,6,7",                # 使用的GPU设备ID（如"0"、"0,1"等）

    # ====== 优化器与学习率 ======
    optimizer='SGD',           # 优化器类型（可选：SGD, Adam, AdamW等）
    lr0=0.01,                  # 初始学习率
    lrf=0.01,                  # 最终学习率因子（最终lr = lr0 * lrf）
    momentum=0.937,            # SGD动量
    weight_decay=0.0005,       # 权重衰减
    warmup_epochs=3,           # 预热轮数
    warmup_momentum=0.8,       # 预热期间的动量
    warmup_bias_lr=0.1,        # 预热期间的偏置学习率
    patience=50,               # 早停耐心值（多少个epoch无提升则停止训练）

    # ====== 数据增强 ======
    hsv_h=0.015,               # HSV色调增强幅度
    hsv_s=0.7,                 # HSV饱和度增强幅度
    hsv_v=0.4,                 # HSV明度增强幅度
    degrees=0.0,               # 随机旋转角度
    translate=0.1,             # 随机平移
    scale=0.7,                 # 随机缩放比例 默认0.5 我设0.7 利好小目标
    shear=0.0,                 # 随机错切
    perspective=0.0,           # 透视变换
    flipud=0.0,                # 上下翻转概率
    fliplr=0.5,                # 左右翻转概率
    mosaic=0.0,                # Mosaic数据增强概率
    mixup=0.05,                # Mixup数据增强概率 默认0.0，我设为0.05 利好小目标
    copy_paste=0.15,           # Copy-paste数据增强概率 默认0.0，我设为0.15 利好小目标
    close_mosaic=10,           # 最后N个epoch关闭mosaic增强

    # ====== 损失函数相关 ======
    box=7.5,                   # 边界框损失权重
    cls=0.5,                   # 分类损失权重
    dfl=1.5,                   # 分布式回归损失权重

    # ====== 训练控制 ======
    workers=8,                 # dataloader线程数
    save=True,                 # 是否保存模型
    save_period=-1,            # 每N个epoch保存一次模型
    resume=False,              # 是否恢复训练
    amp=False,                 # 是否使用混合精度训练 默认True，我设False
    verbose=True,              # 是否显示详细训练信息
    plots=True,                # 是否生成训练过程图表 默认False，我设True
    exist_ok=False,             # 是否允许覆盖已存在的实验目录 

    # ====== 日志与实验管理 ======
    project='yolov13_UAV_Sheep',  # 项目名称
    name='tokenc3_lsnet_p2',                  # 实验名称

    # ====== 评估与保存格式 ======
    save_json=True,             # 是否保存COCO格式json
    save_txt=True,              # 是否保存YOLO格式txt
    save_conf=True,             # 是否保存置信度

    # ====== 其他参数（可选）======
    rect=False,                 # 是否使用矩形训练
    cache=False,                # 是否缓存图片到内存
    single_cls=False,           # 是否将所有类别视为一个类别
    dropout=0.0,                # dropout概率
    overlap_mask=True,          # 是否允许mask重叠
    mask_ratio=4,               # mask下采样比例
    freeze=None,                # 冻结哪些层（如[0,1,2]）
    seed=42,                    # 随机种子
    deterministic=True,         # 是否使用确定性训练
    val=True,                   # 是否在每个epoch后验证
)

logging.info("Training completed!")
logging.info(f"Training results: {results}")