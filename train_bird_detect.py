
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['NUMEXPR_MAX_THREADS'] = '16'
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'

import sys
sys.path.insert(0, r'C:\Users\HabichtZhang\Desktop\机器视觉')

def main():
    import torch
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    from ultralytics import YOLO
    from ultralytics.models.yolo.detect import DetectionTrainer
    from models.modules.iou_losses import apply_iou_loss
    from models.modules.custom_modules import make_improved_trainer, apply_improvements_to_model

    apply_iou_loss('inner_iou')

    v1_dir = os.path.join(r'C:\Users\HabichtZhang\Desktop\机器视觉\runs\detect', 'bird_detect')
    v2_dir = os.path.join(r'C:\Users\HabichtZhang\Desktop\机器视觉\runs\detect', 'bird_detect_v2')
    v3_dir = os.path.join(r'C:\Users\HabichtZhang\Desktop\机器视觉\runs\detect', 'bird_detect_v3')
    data_yaml = r'C:\Users\HabichtZhang\Desktop\机器视觉\datasets\bird_detect\data.yaml'

    IMPROVEMENTS = {'rgc_elan': False, 'c2tssa': False, 'fpsc': False, 'ema': True, 'edec_head': False}

    def fix_checkpoint(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        changed = False
        if 'optimizer' in ckpt and ckpt['optimizer'] is not None:
            ckpt['optimizer'] = None
            changed = True
        if 'scaler' in ckpt and ckpt['scaler'] is not None:
            ckpt['scaler'] = None
            changed = True
        args = ckpt.get('train_args', {})
        for key in ['end2end', 'rle', 'angle', 'kobj', 'pose']:
            if key in args:
                del args[key]
                changed = True
        if args.get('multi_scale') == 0.0:
            args['multi_scale'] = False
            changed = True
        if changed:
            torch.save(ckpt, ckpt_path)
            print("[Fix] Cleaned checkpoint: " + ckpt_path)
        return changed

    def find_best_pt():
        for d in [v3_dir, v2_dir, v1_dir]:
            p = os.path.join(d, 'weights', 'best.pt')
            if os.path.exists(p):
                return p
        return None

    mode = ""
    if len(sys.argv) > 1:
        mode = sys.argv[1].strip().lower()

    if mode == "resume":
        last_pt = os.path.join(v1_dir, 'weights', 'last.pt')
        if not os.path.exists(last_pt):
            print("[Error] No last.pt found for resume!")
            return
        print("[Bird-Detect] Resume mode: continuing from last.pt")
        fix_checkpoint(last_pt)
        model = YOLO(last_pt)
        model.train(resume=True)

    elif mode == "finetune":
        best_pt = find_best_pt()
        if not best_pt:
            print("[Error] No best.pt found!")
            return
        print(f"[Bird-Detect] Finetune v2 from: {best_pt}")
        trainer_cls = make_improved_trainer(IMPROVEMENTS, 1)
        model = YOLO(best_pt)
        model.train(
            trainer=trainer_cls,
            data=data_yaml,
            epochs=30,
            imgsz=640,
            batch=16,
            device=0,
            project=r'C:\Users\HabichtZhang\Desktop\机器视觉\runs\detect',
            name='bird_detect_v2',
            exist_ok=True,
            workers=2,
            amp=True,
            verbose=True,
            cache=False,
            patience=10,
            save=True,
            save_period=-1,
            lr0=0.0002,
            lrf=0.00002,
            optimizer='AdamW',
            cos_lr=True,
            warmup_epochs=1,
            warmup_momentum=0.5,
            warmup_bias_lr=0.01,
            close_mosaic=5,
            dropout=0.1,
            box=10.0,
            cls=0.8,
            dfl=2.0,
            mosaic=0.3,
            mixup=0.1,
            copy_paste=0.1,
            degrees=5.0,
            translate=0.05,
            scale=0.9,
            shear=0.0,
            flipud=0.05,
            fliplr=0.5,
            hsv_h=0.005,
            hsv_s=0.3,
            hsv_v=0.2,
            auto_augment='randaugment',
            erasing=0.1,
            seed=42,
            deterministic=True,
            val=True,
            plots=True,
        )

    elif mode == "finetune_v3":
        best_pt = find_best_pt()
        if not best_pt:
            print("[Error] No best.pt found!")
            return
        print(f"[Bird-Detect] Finetune v3 (text negatives) from: {best_pt}")
        trainer_cls = make_improved_trainer(IMPROVEMENTS, 1)
        model = YOLO(best_pt)
        model.train(
            trainer=trainer_cls,
            data=data_yaml,
            epochs=20,
            imgsz=640,
            batch=16,
            device=0,
            project=r'C:\Users\HabichtZhang\Desktop\机器视觉\runs\detect',
            name='bird_detect_v3',
            exist_ok=True,
            workers=2,
            amp=True,
            verbose=True,
            cache=False,
            patience=8,
            save=True,
            save_period=-1,
            lr0=0.0001,
            lrf=0.00001,
            optimizer='AdamW',
            cos_lr=True,
            warmup_epochs=1,
            warmup_momentum=0.5,
            warmup_bias_lr=0.005,
            close_mosaic=3,
            dropout=0.1,
            box=10.0,
            cls=1.0,
            dfl=2.0,
            mosaic=0.2,
            mixup=0.05,
            copy_paste=0.05,
            degrees=3.0,
            translate=0.03,
            scale=0.95,
            shear=0.0,
            flipud=0.02,
            fliplr=0.5,
            hsv_h=0.003,
            hsv_s=0.2,
            hsv_v=0.15,
            auto_augment='randaugment',
            erasing=0.05,
            seed=42,
            deterministic=True,
            val=True,
            plots=True,
        )

    elif mode == "finetune_v4":
        best_pt = find_best_pt()
        if not best_pt:
            print("[Error] No best.pt found!")
            return
        print(f"[Bird-Detect] Finetune v4 (COCO negatives) from: {best_pt}")
        trainer_cls = make_improved_trainer(IMPROVEMENTS, 1)
        model = YOLO(best_pt)
        model.train(
            trainer=trainer_cls,
            data=data_yaml,
            epochs=20,
            imgsz=640,
            batch=16,
            device=0,
            project=r'C:\Users\HabichtZhang\Desktop\机器视觉\runs\detect',
            name='bird_detect_v4',
            exist_ok=True,
            workers=2,
            amp=True,
            verbose=True,
            cache=False,
            patience=8,
            save=True,
            save_period=-1,
            lr0=0.00008,
            lrf=0.000008,
            optimizer='AdamW',
            cos_lr=True,
            warmup_epochs=1,
            warmup_momentum=0.5,
            warmup_bias_lr=0.005,
            close_mosaic=3,
            dropout=0.1,
            box=10.0,
            cls=1.5,
            dfl=2.0,
            mosaic=0.15,
            mixup=0.03,
            copy_paste=0.03,
            degrees=2.0,
            translate=0.02,
            scale=0.95,
            shear=0.0,
            flipud=0.02,
            fliplr=0.5,
            hsv_h=0.002,
            hsv_s=0.15,
            hsv_v=0.1,
            auto_augment='randaugment',
            erasing=0.03,
            seed=42,
            deterministic=True,
            val=True,
            plots=True,
        )

    elif mode == "finetune_v5":
        best_pt = find_best_pt()
        if not best_pt:
            print("[Error] No best.pt found!")
            return
        print(f"[Bird-Detect] Finetune v5 (1280 + BirdShapeAttention) from: {best_pt}")
        improvements_v5 = {'rgc_elan': False, 'c2tssa': False, 'fpsc': False, 'ema': True, 'edec_head': False, 'bsa': True}
        trainer_cls = make_improved_trainer(improvements_v5, 1)
        model = YOLO(best_pt)
        model.train(
            trainer=trainer_cls,
            data=data_yaml,
            epochs=30,
            imgsz=1280,
            batch=2,
            device=0,
            project=r'C:\Users\HabichtZhang\Desktop\机器视觉\runs\detect',
            name='bird_detect_v5',
            exist_ok=True,
            workers=0,
            amp=True,
            verbose=True,
            cache=False,
            patience=10,
            save=True,
            save_period=-1,
            lr0=0.0001,
            lrf=0.00001,
            optimizer='AdamW',
            cos_lr=True,
            warmup_epochs=2,
            warmup_momentum=0.5,
            warmup_bias_lr=0.005,
            close_mosaic=5,
            dropout=0.1,
            box=10.0,
            cls=1.2,
            dfl=2.0,
            mosaic=0.2,
            mixup=0.05,
            copy_paste=0.05,
            degrees=3.0,
            translate=0.03,
            scale=0.9,
            shear=0.0,
            flipud=0.03,
            fliplr=0.5,
            hsv_h=0.003,
            hsv_s=0.2,
            hsv_v=0.15,
            auto_augment='randaugment',
            erasing=0.05,
            seed=42,
            deterministic=False,
            val=True,
            plots=True,
        )

    elif mode == "resume_v5":
        v5_dir = os.path.join(r'C:\Users\HabichtZhang\Desktop\机器视觉\runs\detect', 'bird_detect_v5')
        v5_best = os.path.join(v5_dir, 'weights', 'best.pt')
        if not os.path.exists(v5_best):
            print("[Error] No v5 best.pt found!")
            return

        print(f"[Bird-Detect] Resume v5 from EMA weights: {v5_best}")
        ckpt = torch.load(v5_best, map_location='cpu', weights_only=False)
        ema_model = ckpt.get('ema')
        if ema_model is None:
            print("[Error] v5 best.pt has no EMA model (checkpoint corrupted)!")
            return
        ema_sd = ema_model.state_dict()
        print(f"  Loaded EMA state_dict: {len(ema_sd)} keys")

        improvements_v5 = {'rgc_elan': False, 'c2tssa': False, 'fpsc': False, 'ema': True, 'edec_head': False, 'bsa': True}

        class _ResumeV5Trainer(DetectionTrainer):
            def get_model(self, cfg=None, weights=None, verbose=True):
                model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
                if improvements_v5:
                    apply_improvements_to_model(model, improvements_v5, 1)
                model_sd = model.state_dict()
                loaded = 0
                for key in ema_sd:
                    if key in model_sd and ema_sd[key].shape == model_sd[key].shape:
                        model_sd[key] = ema_sd[key].clone()
                        loaded += 1
                model.load_state_dict(model_sd)
                print(f"  Transferred {loaded}/{len(ema_sd)} weights from v5 EMA")
                return model

        model = YOLO('yolo11n.pt')
        model.train(
            trainer=_ResumeV5Trainer,
            data=data_yaml,
            epochs=20,
            imgsz=1280,
            batch=2,
            device=0,
            project=r'C:\Users\HabichtZhang\Desktop\机器视觉\runs\detect',
            name='bird_detect_v5_resume',
            exist_ok=True,
            workers=0,
            amp=True,
            verbose=True,
            cache=False,
            patience=8,
            save=True,
            save_period=-1,
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
            cls=1.2,
            dfl=2.0,
            mosaic=0.15,
            mixup=0.03,
            copy_paste=0.03,
            degrees=2.0,
            translate=0.02,
            scale=0.95,
            shear=0.0,
            flipud=0.02,
            fliplr=0.5,
            hsv_h=0.002,
            hsv_s=0.15,
            hsv_v=0.1,
            auto_augment='randaugment',
            erasing=0.03,
            seed=42,
            deterministic=False,
            val=True,
            plots=True,
        )

    elif mode == "finetune_v6":
        # 小目标鸟检测：从v4权重微调，加入小目标鸟+飞机负样本数据
        v4_best = os.path.join(r'C:\Users\HabichtZhang\Desktop\机器视觉\runs\detect', 'bird_detect_v4', 'weights', 'best.pt')
        if not os.path.exists(v4_best):
            print("[Error] v4 best.pt not found!")
            return
        print(f"[Bird-Detect] Finetune v6 (small target + airplane negatives) from: {v4_best}")
        trainer_cls = make_improved_trainer(IMPROVEMENTS, 1)
        model = YOLO(v4_best)
        model.train(
            trainer=trainer_cls,
            data=data_yaml,
            epochs=30,
            imgsz=640,
            batch=16,
            device=0,
            project=r'C:\Users\HabichtZhang\Desktop\机器视觉\runs\detect',
            name='bird_detect_v6',
            exist_ok=True,
            workers=0,
            amp=True,
            verbose=True,
            cache=False,
            patience=10,
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
            box=12.0,       # 加大box loss权重，强化小目标定位
            cls=2.0,        # 加大cls loss权重，区分鸟vs飞机
            dfl=3.0,        # 加大dfl loss权重，精细边界框
            mosaic=0.3,
            mixup=0.1,
            copy_paste=0.2, # 增加copy_paste帮助小目标学习
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
            erasing=0.1,
            seed=42,
            deterministic=False,
            val=True,
            plots=True,
        )

    else:
        last_pt = os.path.join(v1_dir, 'weights', 'last.pt')
        if os.path.exists(last_pt):
            print("[Bird-Detect] Continuing from " + last_pt)
            fix_checkpoint(last_pt)
            model = YOLO(last_pt)
            model.train(resume=True)
        else:
            print("[Bird-Detect] Training from scratch (single-class bird detection)")
            trainer_cls = make_improved_trainer(IMPROVEMENTS, 1)
            model = YOLO('yolo11n.pt')
            model.train(
                trainer=trainer_cls,
                data=data_yaml,
                epochs=100,
                imgsz=640,
                batch=16,
                device=0,
                project=r'C:\Users\HabichtZhang\Desktop\机器视觉\runs\detect',
                name='bird_detect',
                exist_ok=False,
                workers=2,
                amp=True,
                verbose=True,
                cache=False,
                patience=15,
                save=True,
                save_period=-1,
                lr0=0.001,
                lrf=0.0001,
                optimizer='AdamW',
                cos_lr=True,
                warmup_epochs=3,
                warmup_momentum=0.5,
                warmup_bias_lr=0.05,
                close_mosaic=10,
                dropout=0.1,
                box=10.0,
                cls=0.5,
                dfl=2.0,
                mosaic=0.5,
                mixup=0.2,
                copy_paste=0.3,
                degrees=10.0,
                translate=0.1,
                scale=0.7,
                shear=0.0,
                flipud=0.1,
                fliplr=0.5,
                hsv_h=0.01,
                hsv_s=0.5,
                hsv_v=0.3,
                auto_augment='randaugment',
                erasing=0.2,
                seed=42,
                deterministic=True,
                val=True,
                plots=True,
            )

if __name__ == '__main__':
    main()
