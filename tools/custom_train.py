# Copyright (c) OpenMMLab. All rights reserved.
from mmaction.utils import register_all_modules

register_all_modules(init_default_scope=True)

import torch
import copy
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.optim as optim
from mmengine import track_iter_progress
import numpy as np
import os.path as osp
from mmengine.dataset import BaseDataset
from mmaction.registry import DATASETS, TRANSFORMS, MODELS
from tools.utils import ParamDiffAug
from typing import Dict, Optional, Sequence, Tuple
from mmaction.structures import ActionDataSample
from mmcv.transforms import BaseTransform, to_tensor
import argparse
import os
import os.path as osp
import wandb
from tools.custom_aug import DiffAugment
from tools.v_distill import build_original_dataset
from tools.utils import get_dataset
from mmengine.config import Config, DictAction
from mmengine.runner import Runner

from mmaction.registry import RUNNERS

from mmengine.registry import HOOKS
from mmengine.hooks import Hook

@HOOKS.register_module()
class SaveInitParams(Hook):
    def __init__(self):
        ...
    def before_train(self, runner) -> None:
        runner.logger.info('Saving Initial Params...')
        runner.logger.info(runner.model)
        runner.logger.info(runner.__dict__)
        # trajectory.append([p.detach().cpu() for p in runner.model.parameters()])
        # self.save=False
    # def before_run(self, runner) -> None:
    #     runner.logger.info('Saving Initial Params...')
    #     trajectory.append([p.detach().cpu() for p in runner.parameters()])
    
    
    
@HOOKS.register_module()
class SaveEpochTrajectory(Hook):
    """Check invalid loss hook.
    This hook will regularly check whether the loss is valid
    during training.
    Args:
        interval (int): Checking interval (every k iterations).
            Defaults to 50.
    """
    def __init__(self, interval=5):
        self.interval = interval
        # self.trajectory = trajectory
    def after_train_epoch(self, runner, data_batch=None, outputs=None):
        """All subclasses should override this method, if they need any
        operations after each training iteration.
        Args:
            runner (Runner): The runner of the training process.
            batch_idx (int): The index of the current batch in the train loop.
            data_batch (dict or tuple or list, optional): Data from dataloader.
            outputs (dict, optional): Outputs from model.
        """
        if self.every_n_epochs(runner, self.interval):
            runner.logger.info('Saving Epoch Trajectory...')
            trajectory.append([p.detach().cpu() for p in runner.model.parameters()])

@TRANSFORMS.register_module()
class PackDistillInputs(BaseTransform):

    mapping_table = {
        'gt_bboxes': 'bboxes',
        'gt_labels': 'labels',
    }

    def __init__(
            self,
            require_grad=False,
            collect_keys: Optional[Tuple[str]] = None,
            meta_keys: Sequence[str] = ('img_shape', 'img_key', 'video_id',
                                        'timestamp'),
            algorithm_keys: Sequence[str] = (),
    ) -> None:
        self.require_grad = require_grad
        self.collect_keys = collect_keys
        self.meta_keys = meta_keys
        self.algorithm_keys = algorithm_keys

    def transform(self, results: Dict) -> Dict:
        packed_results = dict()
        if self.collect_keys is not None:
            packed_results['inputs'] = dict()
            for key in self.collect_keys:
                packed_results['inputs'][key] = to_tensor(results[key])
        else:
            if 'imgs' in results:
                imgs = results['imgs']
                if self.require_grad:
                    imgs_syn = to_tensor(imgs)
                    imgs_syn = imgs_syn.detach().requires_grad_(True)
                    packed_results['inputs'] = to_tensor(imgs_syn)
                else:
                    packed_results['inputs'] = to_tensor(imgs)
            else:
                raise ValueError(
                    'Cannot get `imgs`, `keypoint`, `heatmap_imgs`, '
                    '`audios` or `text` in the input dict of '
                    '`PackActionInputs`.')

        data_sample = ActionDataSample()

        if 'label' in results:
            data_sample.set_gt_label(results['label'])

        # Set custom algorithm keys
        for key in self.algorithm_keys:
            if key in results:
                data_sample.set_field(results[key], key)

        # Set meta keys
        img_meta = {k: results[k] for k in self.meta_keys if k in results}
        data_sample.set_metainfo(img_meta)
        packed_results['data_samples'] = data_sample
        return packed_results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(collect_keys={self.collect_keys}, '
        repr_str += f'meta_keys={self.meta_keys})'
        return repr_str





def parse_args():
    parser = argparse.ArgumentParser(description='Train a action recognizer')
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
    parser.add_argument('--ipc', type=int, default=50)
    parser.add_argument('--im', type=int, default=224)

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


#! 在train前替换dataloader！ 
def main():   
    args = parse_args()
    global trajectory
    trajectory = []
    args.device = 'cuda'
    
    # wandb.login(key = 'cab22b56672abca555605b07536a36a2c5c4ef39')
    # wandb.init(
    #         project="vdd",
    #     )
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader, loader_train_dict, class_map, class_map_inv = get_dataset('ucf101',im=args.im)
    # channel = 3
    # num_classes =101
    # im_size=[224,224]
    images_all, indices_class = build_original_dataset(channel, num_classes, dst_train, class_map)

    def get_images(c, n): # get random n images from class c
        idx_shuffle = np.random.permutation(indices_class[c])[:n]
        return images_all[idx_shuffle]

    ''' initialize the synthetic data '''
    label_syn = torch.tensor([np.ones(args.ipc,dtype=np.int_)*i for i in range(num_classes)], dtype=torch.long, requires_grad=False).view(-1)
    print("label_syn shape:", label_syn.shape)

    image_syn = torch.randn(size=(num_classes * args.ipc, channel, im_size[0], im_size[1]), dtype=torch.float)
    
    pix_init = 'real'
    if pix_init == 'real':
        print('initialize synthetic data from random real images')
        for c in range(num_classes):
            image_syn.data[c * args.ipc:(c + 1) * args.ipc] = get_images(c, args.ipc).detach().data
    else:
        print('initialize synthetic data from random noise')
    # print("Image_syn", image_syn[0])
    print("image_syn shape:", image_syn.shape)
    
    image_syn_grad = image_syn.detach().to(args.device).requires_grad_(True)
    # print(image_save.requires_grad, image_save.device, id(image_save), image_syn_grad.requires_grad, image_syn_grad.device, id(image_syn_grad), image_syn.requires_grad, image_syn.device, id(image_syn))
    # syn_lr = syn_lr.detach().to(args.device).requires_grad_(True)
    optimizer_img = torch.optim.SGD([image_syn_grad], lr=0.1, momentum=0.5)
    # optimizer_lr = torch.optim.SGD([syn_lr], lr=args.lr_lr, momentum=0.5)
    optimizer_img.zero_grad()
    
    # return
    eval_labs = label_syn
    with torch.no_grad():
        image_save = image_syn_grad
    image_syn_eval, label_syn_eval = copy.deepcopy(image_save.detach()), copy.deepcopy(eval_labs.detach())
    
    best_val_scores = []
    for it_eval in range(1):
        train_pipeline_cfg = [dict(type='PackActionInputs')]
        
        train_dataloader = dict(
            batch_size=32,
            num_workers=8,
            persistent_workers=True,
            sampler=dict(type='DefaultSampler', shuffle=True),
            dataset=dict(
                type='SyntheticDataset',
                image_syn=image_syn_eval.cpu(),
                label_syn = label_syn_eval.cpu(),
                serialize_data=False,
                pipeline=train_pipeline_cfg)
            )

        cfg = Config.fromfile(args.config)

        cfg.train_pipeline = train_pipeline_cfg
        cfg.train_dataloader = train_dataloader

        cfg = merge_args(cfg, args)
        
        runner = RUNNERS.build(cfg)

        # start training
        runner.train()
        
        best_val_score = runner.message_hub.get_info("best_score")
        best_val_scores.append(best_val_score)
    
    best_val_scores = np.array(best_val_scores)
    mean_score = best_val_scores.mean()
    std_score = best_val_scores.std()
    print("Best Val Score: {}({})".format(mean_score,std_score))    

    
if __name__ == '__main__':
    main()
