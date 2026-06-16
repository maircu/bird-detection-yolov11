import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['NUMEXPR_MAX_THREADS'] = '16'
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'

import sys
sys.path.insert(0, r'C:\Users\HabichtZhang\Desktop\机器视觉')

import torch
torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = False

from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
from models.modules.iou_losses import apply_iou_loss
from models.modules.custom_modules import apply_improvements_to_model

apply_iou_loss('inner_iou')

DATA_YAML = r'C:\Users\HabichtZhang\Desktop\机器视觉\datasets\shanghai_33cls\data.yaml'
PROJECT = r'C:\Users\HabichtZhang\Desktop\机器视觉\runs\detect'
SPECIES_BEST = os.path.join(PROJECT, 'ema_1280_33cls', 'weights', 'best.pt')
OCCLUSION_LAST = os.path.join(PROJECT, 'species_occlusion', 'weights', 'last.pt')
OCCLUSION_BEST = os.path.join(PROJECT, 'species_occlusion', 'weights', 'best.pt')
OCCLUSION_V2_BEST = os.path.join(PROJECT, 'species_occlusion_v2', 'weights', 'best.pt')

IMPROVEMENTS = {'rgc_elan': False, 'c2tssa': False, 'fpsc': False, 'ema': True, 'edec_head': False}


def main():
    mode = ""
    if len(sys.argv) > 1:
        mode = sys.argv[1].strip().lower()

    if mode == "gen_occlusion":
        print("[Occlusion] Generating occluded training images...")
        from gen_occlusion_aug import main as gen_main
        gen_main()
        return

    # ---- resume: 从 last.pt 断点续训 ----
    if mode == "resume":
        if not os.path.exists(OCCLUSION_LAST):
            print(f"[Error] Checkpoint not found: {OCCLUSION_LAST}")
            return
        print(f"[Resume] Continuing from {OCCLUSION_LAST}")
        model = YOLO(OCCLUSION_LAST)
        model.train(
            resume=True,
            device=0,
            project=PROJECT,
            name='species_occlusion',
            exist_ok=True,
        )
        return

    # ---- resume_v2: 从 best.pt 的 EMA 权重继续训练(新实验) ----
    if mode == "resume_v2":
        if not os.path.exists(OCCLUSION_BEST):
            print(f"[Error] Best weights not found: {OCCLUSION_BEST}")
            return

        # species_occlusion 闪退导致 model=None，需从 EMA 权重恢复
        ckpt = torch.load(OCCLUSION_BEST, map_location='cpu', weights_only=False)
        src = ckpt.get('model') or ckpt.get('ema')
        if src is None:
            print("[Error] Checkpoint has no model or ema weights!")
            return
        src_sd = src.state_dict() if hasattr(src, 'state_dict') else src
        print(f"[Resume-V2] Loaded {'EMA' if ckpt.get('model') is None else 'model'} weights: {len(src_sd)} keys")

        class _ResumeV2Trainer(DetectionTrainer):
            def get_model(self, cfg=None, weights=None, verbose=True):
                model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
                apply_improvements_to_model(model, IMPROVEMENTS, 33)
                model_sd = model.state_dict()
                loaded = 0
                for key in src_sd:
                    if key in model_sd and src_sd[key].shape == model_sd[key].shape:
                        model_sd[key] = src_sd[key].clone()
                        loaded += 1
                model.load_state_dict(model_sd)
                print(f"  Transferred {loaded}/{len(src_sd)} weights from occlusion model")
                return model

        print("[Resume-V2] Fine-tuning from EMA weights with lower LR")
        model = YOLO('yolo11n.pt')
        model.train(
            trainer=_ResumeV2Trainer,
            data=DATA_YAML,
            epochs=30,
            imgsz=1280,
            batch=2,
            device=0,
            project=PROJECT,
            name='species_occlusion_v2',
            exist_ok=True,
            workers=0,
            amp=True,
            verbose=True,
            cache=False,
            patience=10,
            save=True,
            save_period=5,
            lr0=0.00003,
            lrf=0.000003,
            optimizer='AdamW',
            cos_lr=True,
            warmup_epochs=0,
            close_mosaic=3,
            dropout=0.1,
            box=10.0,
            cls=1.5,
            dfl=2.0,
            mosaic=0.3,
            mixup=0.1,
            copy_paste=0.2,
            degrees=5.0,
            translate=0.05,
            scale=0.85,
            shear=0.0,
            flipud=0.05,
            fliplr=0.5,
            hsv_h=0.005,
            hsv_s=0.3,
            hsv_v=0.2,
            auto_augment='randaugment',
            erasing=0.3,
            seed=42,
            deterministic=False,
            val=True,
            plots=True,
        )
        return

    # ---- finetune_v3: 从 species_occlusion_v2 继续微调，强化品种区分（珠颈斑鸠vs四声杜鹃等） ----
    if mode == "finetune_v3":
        if not os.path.exists(OCCLUSION_V2_BEST):
            print(f"[Error] V2 best weights not found: {OCCLUSION_V2_BEST}")
            return

        ckpt = torch.load(OCCLUSION_V2_BEST, map_location='cpu', weights_only=False)
        src = ckpt.get('model') or ckpt.get('ema')
        if src is None:
            print("[Error] Checkpoint has no model or ema weights!")
            return
        src_sd = src.state_dict() if hasattr(src, 'state_dict') else src
        print(f"[Finetune-V3] Loaded {'EMA' if ckpt.get('model') is None else 'model'} weights: {len(src_sd)} keys")

        class _FinetuneV3Trainer(DetectionTrainer):
            def get_model(self, cfg=None, weights=None, verbose=True):
                model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
                apply_improvements_to_model(model, IMPROVEMENTS, 33)
                model_sd = model.state_dict()
                loaded = 0
                for key in src_sd:
                    if key in model_sd and src_sd[key].shape == model_sd[key].shape:
                        model_sd[key] = src_sd[key].clone()
                        loaded += 1
                model.load_state_dict(model_sd)
                print(f"  Transferred {loaded}/{len(src_sd)} weights from v2 model")
                return model

        print("[Finetune-V3] Fine-tuning from v2 with higher cls weight to improve species discrimination")
        model = YOLO('yolo11n.pt')
        model.train(
            trainer=_FinetuneV3Trainer,
            data=DATA_YAML,
            epochs=20,
            imgsz=1280,
            batch=2,
            device=0,
            project=PROJECT,
            name='species_occlusion_v3',
            exist_ok=True,
            workers=0,
            amp=True,
            verbose=True,
            cache=False,
            patience=8,
            save=True,
            save_period=5,
            lr0=0.00001,
            lrf=0.000001,
            optimizer='AdamW',
            cos_lr=True,
            warmup_epochs=0,
            close_mosaic=3,
            dropout=0.1,
            box=7.0,
            cls=4.0,
            dfl=2.0,
            mosaic=0.3,
            mixup=0.1,
            copy_paste=0.2,
            degrees=5.0,
            translate=0.05,
            scale=0.85,
            shear=0.0,
            flipud=0.05,
            fliplr=0.5,
            hsv_h=0.005,
            hsv_s=0.3,
            hsv_v=0.2,
            auto_augment='randaugment',
            erasing=0.3,
            seed=42,
            deterministic=False,
            val=True,
            plots=True,
        )
        return

    if not os.path.exists(SPECIES_BEST):
        print(f"[Error] Species model not found: {SPECIES_BEST}")
        return

    ckpt = torch.load(SPECIES_BEST, map_location='cpu', weights_only=False)
    src = ckpt.get('model') or ckpt.get('ema')
    if src is None:
        print("[Error] Checkpoint has no model weights!")
        return
    src_sd = src.state_dict() if hasattr(src, 'state_dict') else src
    print(f"Loaded source weights: {len(src_sd)} keys")

    class _OcclusionTrainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
            apply_improvements_to_model(model, IMPROVEMENTS, 33)
            model_sd = model.state_dict()
            loaded = 0
            for key in src_sd:
                if key in model_sd and src_sd[key].shape == model_sd[key].shape:
                    model_sd[key] = src_sd[key].clone()
                    loaded += 1
            model.load_state_dict(model_sd)
            print(f"  Transferred {loaded}/{len(src_sd)} weights from species model")
            return model

    if mode == "train":
        print("[Species-Occlusion] Training with occlusion-augmented data")
        model = YOLO('yolo11n.pt')
        model.train(
            trainer=_OcclusionTrainer,
            data=DATA_YAML,
            epochs=50,
            imgsz=1280,
            batch=2,
            device=0,
            project=PROJECT,
            name='species_occlusion',
            exist_ok=True,
            workers=0,
            amp=True,
            verbose=True,
            cache=False,
            patience=12,
            save=True,
            save_period=5,
            lr0=0.00005,
            lrf=0.000005,
            optimizer='AdamW',
            cos_lr=True,
            warmup_epochs=1,
            warmup_momentum=0.5,
            warmup_bias_lr=0.001,
            close_mosaic=3,
            dropout=0.1,
            box=10.0,
            cls=1.5,
            dfl=2.0,
            mosaic=0.3,
            mixup=0.1,
            copy_paste=0.2,
            degrees=5.0,
            translate=0.05,
            scale=0.85,
            shear=0.0,
            flipud=0.05,
            fliplr=0.5,
            hsv_h=0.005,
            hsv_s=0.3,
            hsv_v=0.2,
            auto_augment='randaugment',
            erasing=0.3,
            seed=42,
            deterministic=False,
            val=True,
            plots=True,
        )

    elif mode == "cleanup":
        print("[Cleanup] Removing occluded augmented images...")
        img_dir = os.path.join(os.path.dirname(DATA_YAML), 'train', 'images')
        lbl_dir = os.path.join(os.path.dirname(DATA_YAML), 'train', 'labels')
        removed = 0
        for f in os.listdir(img_dir):
            if f.startswith('occ'):
                os.remove(os.path.join(img_dir, f))
                removed += 1
        for f in os.listdir(lbl_dir):
            if f.startswith('occ'):
                os.remove(os.path.join(lbl_dir, f))
        print(f"  Removed {removed} augmented images and labels")

    else:
        print("Usage:")
        print("  python train_species_occlusion.py gen_occlusion  - Generate occluded images")
        print("  python train_species_occlusion.py train           - Train from scratch with occlusion data")
        print("  python train_species_occlusion.py resume          - Resume from last.pt (断点续训)")
        print("  python train_species_occlusion.py resume_v2       - Fine-tune from best.pt (新实验, 推荐)")
        print("  python train_species_occlusion.py finetune_v3     - Fine-tune v2 with higher cls weight (强化品种区分)")
        print("  python train_species_occlusion.py cleanup         - Remove augmented images")


if __name__ == '__main__':
    main()
