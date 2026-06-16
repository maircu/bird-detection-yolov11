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
torch.backends.cudnn.deterministic = True

from ultralytics import YOLO
from models.modules.iou_losses import apply_iou_loss
from models.modules.custom_modules import make_improved_trainer

apply_iou_loss('inner_iou')

RUN_DIR = r'C:\Users\HabichtZhang\Desktop\机器视觉\runs\detect\ema_1280_33cls'
LAST_PT = os.path.join(RUN_DIR, 'weights', 'last.pt')
PRETRAINED = r'C:\Users\HabichtZhang\Desktop\机器视觉\runs\detect\ablation_ema\weights\best.pt'
DATA_YAML = r'C:\Users\HabichtZhang\Desktop\机器视觉\datasets\shanghai_33cls\data.yaml'
PROJECT = r'C:\Users\HabichtZhang\Desktop\机器视觉\runs\detect'
NAME = 'ema_1280_33cls'
EPOCHS = 200
IMGSZ = 1280
BATCH = 2
DEVICE = '0'

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
        print(f"[Fix] Cleaned checkpoint: {ckpt_path}")
    return changed


def main():
    if os.path.exists(LAST_PT):
        print(f"[Resume] Found checkpoint: {LAST_PT}")
        fix_checkpoint(LAST_PT)

        ckpt = torch.load(LAST_PT, map_location='cpu', weights_only=False)
        done_epoch = ckpt.get('epoch', 0)
        print(f"[Resume] Trained {done_epoch} epochs, continuing to {EPOCHS}")

        model = YOLO(LAST_PT)
        model.train(resume=True)
    else:
        print("[New] Training from pretrained weights")
        trainer_cls = make_improved_trainer(IMPROVEMENTS, 33)
        model = YOLO(PRETRAINED)
        model.train(
            trainer=trainer_cls,
            data=DATA_YAML,
            epochs=EPOCHS,
            imgsz=IMGSZ,
            batch=BATCH,
            device=DEVICE,
            project=PROJECT,
            name=NAME,
            exist_ok=False,
            workers=2,
            amp=True,
            verbose=True,
            cache=False,
            patience=20,
            save=True,
            save_period=-1,
            lr0=0.001,
            lrf=0.0001,
            optimizer='AdamW',
            cos_lr=True,
            warmup_epochs=5,
            warmup_momentum=0.5,
            warmup_bias_lr=0.05,
            close_mosaic=10,
            dropout=0.1,
            box=10.0,
            cls=0.3,
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
