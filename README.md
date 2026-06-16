# 基于改进YOLOv11的鸟类检测与品种识别系统

## 项目简介

本项目基于YOLOv11n目标检测框架，通过引入EMA多尺度注意力模块、Inner-IoU损失函数和双模型级联架构，实现了面向机场远距离鸟类检测与自然保护区种群统计的自动化鸟类检测与品种识别系统。

**核心指标**：
- 一级鸟检测模型 mAP50 = 0.990
- 二级品种识别模型 mAP50 = 0.856（33类中国常见鸟类）

## 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/maircu/bird-detection-yolov11.git
cd bird-detection-yolov11
```

### 2. 环境配置

```bash
# 创建虚拟环境（推荐）
conda create -n bird python=3.12
conda activate bird

# 安装依赖
pip install -r requirements.txt
```

主要依赖：ultralytics>=8.3, opencv-python, Pillow, numpy, torch

### 3. 视频检测

将待检测视频放入 `video/` 文件夹，运行：

```bash
python detect.py
```

运行后选择模式 1（视频检测），然后选择视频文件即可。

视频检测选项：
- **Crop模式**：裁剪二次验证，遮挡场景品种识别更准
- **Slice模式**：SAHI切片推理，远距离小目标检测更强
- **运动检测**：MOG2背景减除，过滤静止误检

### 4. 图片检测

将待检测图片放入 `photo/` 文件夹，运行：

```bash
python detect.py
```

选择模式 2（图片检测），自动使用ultra-fine切片模式检测所有图片，结果保存到 `output/photos/`。

### 5. 模型训练

训练需要自行准备数据集（见下方数据集说明）。

```bash
# 一级鸟检测模型训练
python train_bird_detect.py finetune_v4

# 二级品种识别模型训练（含遮挡增强）
python train_species_occlusion.py train

# 从已有权重继续微调
python train_species_occlusion.py finetune_v3
```

## 项目结构

```
bird-detection-yolov11/
├── detect.py                    # 主检测脚本（视频+图片）
├── detect_zhujing.py            # 珠颈斑鸠专用检测脚本
├── train_bird_detect.py         # 鸟检测模型训练脚本
├── train_species_occlusion.py   # 品种识别遮挡增强训练脚本
├── train_1280.py                # 1280分辨率训练脚本
├── train_shanghai.py            # 上海鸟类训练脚本
├── merge_bird_dataset.py        # 合并CUB-200+上海鸟类数据集
├── gen_occlusion_aug.py         # 遮挡增强图片生成
├── gen_small_bird_aug.py        # 小目标剪影数据增强
├── gen_text_negatives.py        # 文字负样本生成器
├── hard_negative_mining.py      # 难负样本挖掘
├── coco_negative_pipeline.py    # COCO负样本处理管道
├── extract_no_bird.py           # 从视频提取无鸟帧
├── requirements.txt             # Python依赖
│
├── models/                      # 模型配置与自定义模块
│   ├── yolo11n-improved.yaml    # 改进模型配置（EMA+Inner-IoU）
│   └── modules/
│       ├── custom_modules.py    # EMA等自定义模块
│       └── iou_losses.py        # Inner-IoU损失函数
│
├── runs/detect/                 # 模型权重
│   ├── bird_detect_v4/weights/best.pt      # 一级检测模型（mAP50=0.990）
│   └── species_occlusion_v2/weights/best.pt # 二级品种识别模型（mAP50=0.856）
│
├── 结果/                        # 检测结果示例图片
├── video/                       # 测试视频（放入mp4/avi文件后使用）
├── photo/                       # 图片检测输入（放入jpg/png文件后使用）
└── output/                      # 检测输出（自动生成）
```

## 数据集说明

本项目使用的数据集未包含在仓库中，需自行下载：

1. **CUB-200-2011**：[下载地址](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html)，细粒度鸟类分类数据集，200种11788张图像
2. **上海鸟类数据集**：区域性鸟类数据集，需自行采集或获取
3. **COCO 2017**：[下载地址](https://cocodataset.org/)，用于筛选猫/狗/人负样本

数据集目录结构：
```
datasets/
├── bird_detect/        # 一级检测数据集（单类bird）
├── shanghai_33cls/     # 二级品种识别数据集（33类）
└── birds/              # 原始鸟类数据集
```

## 核心技术

| 技术 | 说明 | 效果 |
|------|------|------|
| EMA注意力 | 分组1D池化+跨空间融合 | 小目标mAP +60.0% |
| Inner-IoU | 辅助内缩框+焦点加权 | mAP50-95 +3% |
| 双模型级联 | 检测与识别解耦 | 检测0.990/识别0.856 |
| 遮挡增强 | 9083张遮挡样本 | mAP50 +3.2% |
| 负样本训练 | COCO+文字+无鸟帧+飞机 | 误检大幅下降 |
| SAHI切片 | 640切片+批量推理 | 小目标检出提升 |
| Cascade-NMS | 品种优先去重 | 消除重复标注 |

## 模型权重

项目已包含训练好的模型权重，可直接使用：
- 一级检测：`runs/detect/bird_detect_v4/weights/best.pt`
- 二级识别：`runs/detect/species_occlusion_v2/weights/best.pt`

## 33类品种列表

白顶鹎、白鹭、白眉鸭、斑嘴鸭、北红尾鸲、苍鹭、赤颈鸭、大山雀、反嘴鹬、黑翅长脚鹬、黑水鸡、鹤鹬、红胁蓝尾鸲、黄腰柳莺、环颈鸻、灰头鹀、矶鹬、罗纹鸭、绿翅鸭、绿头鸭、琵嘴鸭、普通鸬鹚、山斑鸠、四声杜鹃、乌鸫、喜鹊、燕雀、夜鹭、银喉长尾山雀、鸳鸯、针尾鸭、珠颈斑鸠、棕背伯劳

## 注意事项

- 训练1280分辨率时显存需求较大，建议batch=2, workers=0
- 切片推理模式速度约8-10fps，Fast模式约15fps
- 品种识别覆盖33类中国常见鸟类，未覆盖品种标注为"鸟"
- 预训练权重 `yolo11n.pt` 需从Ultralytics下载，首次运行会自动下载
