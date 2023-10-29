CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/recognition/tsn/tsn_ucf101-32x2x1.py --seed=0 &
CUDA_VISIBLE_DEVICES=1 python tools/train.py configs/recognition/tsn/tsn_ucf101-8x8x1.py --seed=0 &
CUDA_VISIBLE_DEVICES=2 python tools/train.py configs/recognition/tsn/tsn_ucf101-16x1x1.py --seed=0 &
CUDA_VISIBLE_DEVICES=3 python tools/train.py configs/recognition/tsn/tsn_ucf101-16x4x1.py --seed=0 
# CUDA_VISIBLE_DEVICES=4 python tools/train.py configs/recognition/tsn/tsn_ucf101-1x1x8.py --seed=0

