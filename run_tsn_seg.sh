# CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/recognition/tsn/tsn_ucf101-1x1x5.py --seed=0 &
CUDA_VISIBLE_DEVICES=5 python tools/train.py configs/recognition/tsn/tsn_ucf101-1x1x7.py --seed=0 &
# CUDA_VISIBLE_DEVICES=4 python tools/train.py configs/recognition/tsn/tsn_ucf101-1x1x8.py --seed=0 &
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/recognition/tsn/tsn_ucf101-1x1x9.py --seed=0 &
# CUDA_VISIBLE_DEVICES=2 python tools/train.py configs/recognition/tsn/tsn_ucf101-1x1x1.py --seed=0 &
# CUDA_VISIBLE_DEVICES=1 python tools/train.py configs/recognition/tsn/tsn_ucf101-1x1x3.py --seed=0

