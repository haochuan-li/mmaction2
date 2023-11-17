import os
import argparse
import torch
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import torch.distributed as dist

# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp

from mmengine.config import Config, DictAction
from mmengine.runner import Runner

from mmaction.registry import RUNNERS
from mmengine.registry import HOOKS
from mmengine.hooks import Hook

@HOOKS.register_module()
class SaveInitParams(Hook):
    def __init__(self) -> None:
        ...
    def before_train(self, runner) -> None:
        # try:
        rank = dist.get_rank()
        # except:
        #     rank = 0
        if rank == 0:
            runner.logger.info('Saving Initial Params...')
            timestamps.append([p.detach().cpu() for p in runner.model.parameters()])



@HOOKS.register_module()
class SaveEpochTrajectory(Hook):
    def __init__(self, interval=5):
        self.interval = interval
    def after_train_epoch(self, runner, data_batch=None, outputs=None):
        """All subclasses should override this method, if they need any
        operations after each training iteration.
        Args:
            runner (Runner): The runner of the training process.
            batch_idx (int): The index of the current batch in the train loop.
            data_batch (dict or tuple or list, optional): Data from dataloader.
            outputs (dict, optional): Outputs from model.
        """
        # try:
        rank = dist.get_rank()
        # except:
        #     rank = 0
        if rank == 0 and self.every_n_epochs(runner, self.interval):
            runner.logger.info('Saving Epoch Trajectory...')
            timestamps.append([p.detach().cpu() for p in runner.model.parameters()])


def parse_args():
    parser = argparse.ArgumentParser(description='Train a action recognizer')
    
    parser.add_argument('--dataset', type=str, default='ucf101', help='dataset')
    # parser.add_argument('--subset', type=str, default='imagenette', help='subset')
    parser.add_argument('--model', type=str, default='tsn', help='model')
    parser.add_argument('--buffer_path', type=str, default='/data/haochuan/buffers', help='buffer path')
    parser.add_argument('--save_interval', type=int, default=10)
    parser.add_argument('--num_experts', type=int, default=100, help='training iterations')
    
    parser.add_argument('config', help='train config file path')
    
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume',
        nargs='?',
        type=str,
        const='auto',
        help='If specify checkpoint path, resume from it, while if not '
        'specify, try to auto resume from the latest checkpoint '
        'in the work directory.')
    parser.add_argument(
        '--amp',
        action='store_true',
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    parser.add_argument(
        '--auto-scale-lr',
        action='store_true',
        help='whether to auto scale the learning rate according to the '
        'actual batch size and the original batch size.')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--diff-rank-seed',
        action='store_true',
        help='whether or not set different seeds for different ranks')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def merge_args(cfg, args):
    """Merge CLI arguments to config."""
    if args.no_validate:
        cfg.val_cfg = None
        cfg.val_dataloader = None
        cfg.val_evaluator = None

    cfg.launcher = args.launcher

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    # enable automatic-mixed-precision training
    if args.amp is True:
        optim_wrapper = cfg.optim_wrapper.get('type', 'OptimWrapper')
        assert optim_wrapper in ['OptimWrapper', 'AmpOptimWrapper'], \
            '`--amp` is not supported custom optimizer wrapper type ' \
            f'`{optim_wrapper}.'
        cfg.optim_wrapper.type = 'AmpOptimWrapper'
        cfg.optim_wrapper.setdefault('loss_scale', 'dynamic')

    # resume training
    if args.resume == 'auto':
        cfg.resume = True
        cfg.load_from = None
    elif args.resume is not None:
        cfg.resume = True
        cfg.load_from = args.resume

    # enable auto scale learning rate
    if args.auto_scale_lr:
        cfg.auto_scale_lr.enable = True

    # set random seeds
    if cfg.get('randomness', None) is None:
        cfg.randomness = dict(
            seed=args.seed,
            diff_rank_seed=args.diff_rank_seed,
            deterministic=args.deterministic)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    return cfg


def main():
    # try:
    # rank = dist.get_rank()
    # except:
#     rank = 0
    #     print('No distributed setting')
    
    custom_hooks = [
        dict(type='SaveEpochTrajectory', interval=1),
        dict(type='SaveInitParams')
        # dict(type='Test')
    ]
    
    args = parse_args()
    
    save_dir = os.path.join(args.buffer_path, args.dataset)
    save_dir = os.path.join(save_dir, args.model)
    
    # common settings for experts
    cfg = Config.fromfile(args.config)
    # merge cli arguments to config
    cfg = merge_args(cfg, args)
    # add trajectory save hook
    cfg['custom_hooks'] = custom_hooks
    
    global timestamps
    trajectories = []
    
    for it in tqdm(range(args.num_experts),unit="exp"):
        timestamps = []
        # set different initial params
        cfg.randomness = dict(
                        seed=int(time.time() * 1000) % 100000,
                        diff_rank_seed=args.diff_rank_seed,
                        deterministic=args.deterministic)
        
        # will save the params before training and after each epoch
        runner = RUNNERS.build(cfg)
        
        runner.train()
        
        rank = dist.get_rank()
        if rank == 0:
            print("Expert {} seed:{}".format(it+1, cfg.randomness['seed']))
        
        # add expert trajectory
        if rank == 0:
            trajectories.append(timestamps)

        if len(trajectories) == args.save_interval and rank == 0:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            n = 0
            while os.path.exists(os.path.join(save_dir, "replay_buffer_{}.pt".format(n))):
                n += 1
            print("Saving {}".format(os.path.join(save_dir, "replay_buffer_{}.pt".format(n))))
            torch.save(trajectories, os.path.join(save_dir, "replay_buffer_{}.pt".format(n)))
            trajectories = []
    
if __name__ == '__main__':
    main()


