# 基于改进YOLOv11的鸟类检测与品种识别系统

## 项目简介

本项目基于YOLOv11n目标检测框架，通过引入EMA多尺度注意力模块、Inner-IoU损失函数和双模型级联架构，实现了面向机场远距离鸟类检测与自然保护区种群统计的自动化鸟类检测与品种识别系统。

**核心指标**：
- 一级鸟检测模型 mAP50 = 0.990
- 二级品种识别模型 mAP50 = 0.856（33类中国常见鸟类）

## 项目结构

```
机器视觉/
├── detect.py                    # 主检测脚本（视频+图片，三种模式）
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
├── generate_report_v2.py        # 报告生成脚本
├── requirements.txt             # Python依赖
├── yolo11n.pt                   # YOLOv11n预训练权重
│
├── models/                      # 模型配置与自定义模块
│   ├── yolo11n-improved.yaml    # 改进模型配置（EMA+Inner-IoU）
│   └── modules/
│       ├── custom_modules.py    # BSA等自定义模块
│       └── iou_losses.py        # Inner-IoU损失函数
│
├── runs/detect/                 # 训练结果（每个子目录对应一个模型）
│   ├── bird_detect_v4/          # ⭐ 一级检测模型（mAP50=0.990）
│   ├── species_occlusion_v2/    # ⭐ 二级品种识别模型（mAP50=0.856）
│   ├── bird_detect_v6/          # 小目标增强检测模型
│   ├── ablation_ema/            # EMA消融实验
│   ├── ablation_fpsc/           # FPSC消融实验
│   ├── distill_ema/             # 知识蒸馏实验
│   └── ...                      # 其他历史模型
│
├── datasets/                    # 数据集
│   ├── bird_detect/             # 一级检测数据集（单类bird）
│   ├── shanghai_33cls/          # 二级品种识别数据集（33类）
│   └── birds/                   # 原始鸟类数据集
│
├── video/                       # 测试视频
├── photo/                       # 图片检测输入
├── output/                      # 检测输出
│   └── 暂时能用/                # 可用的检测输出视频
│
├── 可视化/                      # 训练可视化图片
├── _archive/                    # 已归档的旧脚本
│
├── 报告/                        # 项目报告
│   ├── report.tex               # LaTeX源文件
│   ├── figs/                    # 报告图片
│   ├── 模板/                    # 报告模板与校徽校名
│   └── report_figures.html      # 报告配图生成页面
│
├── ppt_diagrams.html            # PPT示意图
├── model_results_v2.csv         # 模型训练效果汇总表
└── 模型训练效果汇总.html         # 可视化HTML报告
```

## 快速开始

### 环境配置

```bash
pip install -r requirements.txt
```

主要依赖：ultralytics, opencv-python, Pillow, numpy, torch

### 视频检测

```bash
python detect.py
```

运行后选择模式：
1. **Fast模式**（~15fps）：标准检测+品种识别
2. **+Crop模式**（~8fps）：增加裁剪二次验证，遮挡场景更准
3. **+Slice模式**（~8-10fps）：SAHI切片推理，小目标检测更强

### 图片检测

将图片放入 `photo/` 文件夹，运行detect.py选择图片模式即可。

### 模型训练

```bash
# 一级鸟检测模型训练
python train_bird_detect.py finetune_v4

# 二级品种识别模型训练（含遮挡增强）
python train_species_occlusion.py train

# 从已有权重继续训练
python train_bird_detect.py resume
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

当前使用的模型权重：
- 一级检测：`runs/detect/bird_detect_v4/weights/best.pt`
- 二级识别：`runs/detect/species_occlusion_v2/weights/best.pt`

## 注意事项

- 训练1280分辨率时显存需求较大，建议batch=2, workers=0
- 切片推理模式速度约8-10fps，Fast模式约15fps
- 品种识别覆盖33类中国常见鸟类，未覆盖品种标注为"鸟"
- 报告LaTeX文件需使用XeLaTeX编译
