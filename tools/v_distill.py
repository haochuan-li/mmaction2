from mmaction.utils import register_all_modules
register_all_modules(init_default_scope=True)

import os
import os.path as osp
import argparse
import numpy as np
import torch
import torch.nn as nn
import torchvision.utils
from mmengine.config import Config, DictAction
from tools.utils import get_dataset, get_eval_pool, evaluate_synset, get_time, DiffAugment, ParamDiffAug, build_original_dataset
import wandb
import copy
import random
from mmengine.runner import Runner
from mmaction.structures import ActionDataSample
from tools.custom_dataset import SyntheticDataset
from tools.custom_aug import DiffAug
from mmaction.registry import RUNNERS, MODELS
from tools.reparam_module import ReparamModule

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

networks = dict(
            alexnet = dict(
            type='Recognizer2D',
            backbone=dict(
                pretrained=True,
                type="torchvision.alexnet",
                ),
            cls_head=dict(
                type='TSNHead',
                num_classes=101,
                in_channels=256,
                spatial_type='avg',
                consensus=dict(type='AvgConsensus', dim=1),
                dropout_ratio=0.4,
                init_std=0.01,
                average_clips='prob'),
            data_preprocessor=dict(
                type='ActionDataPreprocessor',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                format_shape='NCHW'),
            train_cfg=None,
            test_cfg=None)
            ,
            resnet = dict(
            type='Recognizer2D',
            backbone=dict(
                type='ResNet',
                pretrained='https://download.pytorch.org/models/resnet50-11ad3fa6.pth',
                depth=50,
                norm_eval=False),
            cls_head=dict(
                type='TSNHead',
                num_classes=101,
                in_channels=2048,
                spatial_type='avg',
                consensus=dict(type='AvgConsensus', dim=1),
                dropout_ratio=0.4,
                init_std=0.01,
                average_clips='prob'),
            data_preprocessor=dict(
                type='ActionDataPreprocessor',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                format_shape='NCHW'),
            train_cfg=None,
            test_cfg=None)
            ,
            convnet = dict(
            type='Recognizer2D',
            backbone=dict(
                type="ConvNet",
                channel=3,
                num_classes=101,
                net_act='relu',
                net_norm='batchnorm',
                net_width=128,
                net_depth=5,
                net_pooling='avgpooling'
            ),
            cls_head=dict(
                type='TSNHead',
                num_classes=101,
                in_channels=128,
                spatial_type='avg',
                consensus=dict(type='AvgConsensus', dim=1),
                dropout_ratio=0.4,
                init_std=0.01,
                average_clips='prob'),
            data_preprocessor=dict(
                type='ActionDataPreprocessor',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                format_shape='NCHW'),
            train_cfg=None,
            test_cfg=None
            )
)

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


def main(args):
    if args.max_experts is not None and args.max_files is not None:
        args.total_experts = args.max_experts * args.max_files

    print("CUDNN STATUS: {}".format(torch.backends.cudnn.enabled))

    args.dsa = True if args.dsa == 'True' else False
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    accs_all_exps = dict() # record performances of all experiments
    model_eval_pool = ['tsn']
    for key in model_eval_pool:
        accs_all_exps[key] = []

    args.dc_aug_param = None

    args.dsa_param = ParamDiffAug()

    dsa_params = args.dsa_param
    if args.zca:
        zca_trans = args.zca_trans
    else:
        zca_trans = None
    
    wandb.login(key='cab22b56672abca555605b07536a36a2c5c4ef39')
    wandb.login()
    wandb.init(sync_tensorboard=False,
               project="VideoDataDistillation",
               config=args,
               )
    
    if args.skip_first_eva==False:
        eval_it_pool = np.arange(0, args.Iteration + 1, args.eval_it).tolist()
    else:
        eval_it_pool = np.arange(args.eval_it, args.Iteration + 1, args.eval_it).tolist()
        
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader, loader_train_dict, class_map, class_map_inv = get_dataset(args.dataset, args.data_path, args.batch_real, args.subset, args=args, im=args.im)
    
    print('Hyper-parameters: \n', args.__dict__)
 
    args = type('', (), {})()

    for key in wandb.config._items:
        setattr(args, key, wandb.config._items[key])

    args.dsa_param = dsa_params
    args.zca_trans = zca_trans

    args.distributed = torch.cuda.device_count() > 1
    
    if args.batch_syn is None:
        args.batch_syn = num_classes * args.ipc

    images_all, indices_class = build_original_dataset(channel, num_classes, dst_train, class_map)
    
    def get_images(c, n): # get random n images from class c
        idx_shuffle = np.random.permutation(indices_class[c])[:n]
        return images_all[idx_shuffle]
    
    # ''' initialize the synthetic data '''
    label_syn = torch.cat([torch.ones(args.ipc,dtype=torch.long)*i for i in range(num_classes)]) # [0,0,0, 1,1,1, ..., 9,9,9]

    image_syn = torch.randn(size=(num_classes * args.ipc, channel, im_size[0], im_size[1]), dtype=torch.float)

    syn_lr = torch.tensor(args.lr_teacher).to(args.device)
    
    if args.pix_init == 'real':
        print('initialize synthetic data from random real images')
        for c in range(num_classes):
            image_syn.data[c * args.ipc:(c + 1) * args.ipc] = get_images(c, args.ipc).detach().data
    else:
        print('initialize synthetic data from random noise')
    # print("image_syn shape:", image_syn.shape)
    
    ''' training '''
    image_syn = image_syn.detach().to(args.device).requires_grad_(True)
    syn_lr = syn_lr.detach().to(args.device).requires_grad_(True)
    optimizer_img = torch.optim.SGD([image_syn], lr=args.lr_img, momentum=0.5)
    optimizer_lr = torch.optim.SGD([syn_lr], lr=args.lr_lr, momentum=0.5)
    criterion = nn.CrossEntropyLoss()
    
    optimizer_img.zero_grad()

    print('%s training begins'%get_time())

    expert_dir = os.path.join(args.buffer_path, args.model)

    print("Expert Dir: {}".format(expert_dir))

    if args.load_all:
        buffer = []
        n = 0
        while os.path.exists(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n))):
            buffer = buffer + torch.load(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n)))
            n += 1
        if n == 0:
            raise AssertionError("No buffers detected at {}".format(expert_dir))
    else:
        expert_files = []
        n = 0
        while os.path.exists(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n))):
            expert_files.append(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n)))
            n += 1
        if n == 0:
            raise AssertionError("No buffers detected at {}".format(expert_dir))
        file_idx = 0
        expert_idx = 0
        random.shuffle(expert_files)
        if args.max_files is not None:
            expert_files = expert_files[:args.max_files]
        print("loading file {}".format(expert_files[file_idx]))
        buffer = torch.load(expert_files[file_idx])
        if args.max_experts is not None:
            buffer = buffer[:args.max_experts]
        random.shuffle(buffer)
    
    track_val_scores = []
    # """
    #? Replace with Runner
    for it in range(0, args.Iteration+1):
        save_this_it = False

        # wandb.log({"Progress": it}, step=it)

        ''' Evaluate synthetic data '''
        if it in eval_it_pool:
            best_val_scores = []
            for model_eval in model_eval_pool:
                print('-------------------------\nEvaluation\nmodel_train = %s, model_eval = %s, iteration = %d'%(args.model, model_eval, it))
                if args.dsa:
                    print('DSA augmentation strategy: \n', args.dsa_strategy)
                    print('DSA augmentation parameters: \n', args.dsa_param.__dict__)
                else:
                    print('DC augmentation parameters: \n', args.dc_aug_param)

                eval_labs = label_syn
                with torch.no_grad():
                    image_save = image_syn
                image_syn_eval, label_syn_eval = copy.deepcopy(image_save.detach()), copy.deepcopy(eval_labs.detach()) # avoid any unaware modification

                args.lr_net = syn_lr.item()

                for it_eval in range(args.num_eval):
                    # train_pipeline_cfg = [dict(type='DiffAug', strategy='color_crop_cutout_flip_scale_rotate'), dict(type='PackActionInputs')]
                    train_pipeline_cfg = [
                                            # dict(
                                            #     type='MultiScaleCrop',
                                            #     input_size=224,
                                            #     scales=(1, 0.875, 0.75, 0.66),
                                            #     random_crop=False,
                                            #     max_wh_scale_gap=1),
                                            # dict(type='Flip', flip_ratio=0.5), 
                                            dict(type='PackActionInputs')]
                    
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
                wandb.log({"Best Val Score": mean_score, 
                           "Best Val Std": std_score})
                if len(track_val_scores)==0 or mean_score > track_val_scores[-1]:
                    print("Save This Iter!")
                    save_this_it = True
                    track_val_scores.append(mean_score)
                
        if save_this_it or it % 1000 == 0:
            with torch.no_grad():
                image_save = image_syn.cuda()

                save_dir = os.path.join(".", "logged_files", args.dataset, wandb.run.name)

                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                image_save_name = "images_" if not save_this_it else "images_best_"
                label_save_name = "labels_" if not save_this_it else "labels_best_"
                
                torch.save(image_save.cpu(), os.path.join(save_dir, image_save_name+"{}.pt".format(it)))
                torch.save(label_syn.cpu(), os.path.join(save_dir, label_save_name+"{}.pt".format(it)))
                
                wandb.log({"Pixels": wandb.Histogram(torch.nan_to_num(image_syn.detach().cpu()))}, step=it)
                if save_this_it:
                    save_this_it = False
              
        wandb.log({"Synthetic_LR": syn_lr.detach().cpu()})
        
        model_cfg = networks['convnet']
        
        student_net = MODELS.build(copy.deepcopy(model_cfg)).to(args.device)
        
        student_net = ReparamModule(student_net)

        if args.distributed:
            student_net = torch.nn.DataParallel(student_net)

        student_net.train()

        num_params = sum([np.prod(p.size()) for p in (student_net.parameters())])

        if args.load_all:
            expert_trajectory = buffer[np.random.randint(0, len(buffer))]
        else:
            expert_trajectory = buffer[expert_idx]
            expert_idx += 1
            if expert_idx == len(buffer):
                expert_idx = 0
                file_idx += 1
                if file_idx == len(expert_files):
                    file_idx = 0
                    random.shuffle(expert_files)
                print("loading file {}".format(expert_files[file_idx]))
                if args.max_files != 1:
                    del buffer
                    buffer = torch.load(expert_files[file_idx])
                if args.max_experts is not None:
                    buffer = buffer[:args.max_experts]
                random.shuffle(buffer)

        start_epoch = np.random.randint(0, args.max_start_epoch)
        starting_params = expert_trajectory[start_epoch]
        
        target_params = expert_trajectory[start_epoch+args.expert_epochs]
        target_params = torch.cat([p.data.to(args.device).reshape(-1) for p in target_params], 0)

        student_params = [torch.cat([p.data.to(args.device).reshape(-1) for p in starting_params], 0).requires_grad_(True)]

        starting_params = torch.cat([p.data.to(args.device).reshape(-1) for p in starting_params], 0)

        syn_images = image_syn

        y_hat = label_syn.to(args.device)

        param_loss_list = []
        param_dist_list = []
        indices_chunks = []
        
        for step in range(args.syn_steps):
            wandb.watch(
                    models = student_net,
                    log='all',
                    log_freq = 5)
            
            if not indices_chunks:
                indices = torch.randperm(len(syn_images))
                indices_chunks = list(torch.split(indices, args.batch_syn))

            these_indices = indices_chunks.pop()

            x = syn_images[these_indices]
            this_y = y_hat[these_indices]

            if args.dsa and (not args.no_aug):
                x = DiffAugment(x, args.dsa_strategy, param=args.dsa_param)

            tensors = [t.unsqueeze(0) for t in x]
            labels = []
            for l in this_y:
                data_sample = ActionDataSample()
                data_sample.set_gt_label(l)
                labels.append(data_sample)
            
            batch = {"inputs": tensors, "data_samples":labels}
            # data = {"inputs": x.unsqueeze(1), "data_samples": labels}
            
            if args.distributed:
                forward_params = student_params[-1].unsqueeze(0).expand(torch.cuda.device_count(), -1)
                data = student_net.module.module.data_preprocessor(batch, training=True)
            else:
                forward_params = student_params[-1]
                data = student_net.module.data_preprocessor(batch, training=True)
                # data = student_net.data_preprocessor(batch, training=True)

            # logits = student_net(**data, flat_param=forward_params, mode='tensor')
            # x = student_net.module.module.cls_head(logits, 1)
            # ce_loss = criterion(x, this_y)

            logits = student_net(**data, flat_param=forward_params, mode='logit')
            ce_loss = criterion(logits, this_y)
            
            grad = torch.autograd.grad(ce_loss, student_params[-1], create_graph=True)[0]
            student_params.append(student_params[-1] - syn_lr * grad)

            if step%5 == 0:
                print("Metrics:", ce_loss, "Grad:",grad)
            
            wandb.log({"student ce loss": ce_loss, 
                       "internal grad":grad.detach().cpu()})

        param_loss = torch.tensor(0.0).to(args.device)
        param_dist = torch.tensor(0.0).to(args.device)

        param_loss += torch.nn.functional.mse_loss(student_params[-1], target_params, reduction="sum")
        param_dist += torch.nn.functional.mse_loss(starting_params, target_params, reduction="sum")

        param_loss_list.append(param_loss)
        param_dist_list.append(param_dist)


        param_loss /= num_params
        param_dist /= num_params

        param_loss /= param_dist

        grand_loss = param_loss

        optimizer_img.zero_grad()
        optimizer_lr.zero_grad()

        grand_loss.backward()

        wandb.log({"Grand_Loss": grand_loss.detach().cpu(),
                   "image_syn":image_syn.detach().cpu(),
                   "image_syn_grad": image_syn.grad,
                   })
        
        if it%10 == 0:
            print('%s iter = %04d, loss = %.4f' % (get_time(), it, grand_loss.item()))
            print("Image Grad:",image_syn.grad)

        optimizer_img.step()
        optimizer_lr.step()

        for _ in student_params:
            del _

    print("Best Val Syn Image Score:", best_val_scores)
    
    wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--dataset', type=str, default='ucf101', help='dataset')

    parser.add_argument('--subset', type=str, default='imagenette', help='ImageNet subset. This only does anything when --dataset=ImageNet')

    parser.add_argument('--model', type=str, default='ConvNet', help='model')

    parser.add_argument('--ipc', type=int, default=1, help='image(s) per class')
   
    parser.add_argument('--im', type=int, default=224)
    
    parser.add_argument('--eval_mode', type=str, default='S',
                        help='eval_mode, check utils.py for more info')

    parser.add_argument('--num_eval', type=int, default=5, help='how many networks to evaluate on')

    parser.add_argument('--eval_it', type=int, default=100, help='how often to evaluate')

    parser.add_argument('--epoch_eval_train', type=int, default=1000, help='epochs to train a model with synthetic data')
    # distillation step
    parser.add_argument('--Iteration', type=int, default=5000, help='how many distillation steps to perform')

    parser.add_argument('--lr_img', type=float, default=1000, help='learning rate for updating synthetic images')
    parser.add_argument('--lr_lr', type=float, default=1e-05, help='learning rate for updating... learning rate')
    parser.add_argument('--lr_teacher', type=float, default=0.01, help='initialization for synthetic learning rate')

    parser.add_argument('--lr_init', type=float, default=0.01, help='how to init lr (alpha)')

    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
    parser.add_argument('--batch_syn', type=int, default=None, help='should only use this if you run out of VRAM')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')

    parser.add_argument('--pix_init', type=str, default='real', choices=["noise", "real"],
                        help='noise/real: initialize synthetic images from random noise or randomly sampled real images.')

    parser.add_argument('--dsa', type=str, default='True', choices=['True', 'False'],
                        help='whether to use differentiable Siamese augmentation.')

    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate',
                        help='differentiable Siamese augmentation strategy')

    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument('--buffer_path', type=str, default='./buffers/tsn_50exp', help='buffer path')

    # M
    parser.add_argument('--expert_epochs', type=int, default=3, help='how many expert epochs the target params are')
    # N
    parser.add_argument('--syn_steps', type=int, default=20, help='how many steps to take on synthetic data')
    # T_max
    parser.add_argument('--max_start_epoch', type=int, default=25, help='max epoch we can start at')

    parser.add_argument('--zca', action='store_true', help="do ZCA whitening")

    parser.add_argument('--load_all', action='store_true', help="only use if you can fit all expert trajectories into RAM")

    parser.add_argument('--no_aug', type=bool, default=False, help='this turns off diff aug during distillation')
    parser.add_argument('--skip_first_eva', action='store_true', help='this turns off skip first eval')

    parser.add_argument('--texture', action='store_true', help="will distill textures instead")
    parser.add_argument('--canvas_size', type=int, default=2, help='size of synthetic canvas')
    parser.add_argument('--canvas_samples', type=int, default=1, help='number of canvas samples per iteration')


    parser.add_argument('--max_files', type=int, default=None, help='number of expert files to read (leave as None unless doing ablations)')
    parser.add_argument('--max_experts', type=int, default=None, help='number of experts to read per file (leave as None unless doing ablations)')

    parser.add_argument('--force_save', action='store_true', help='this will save images for 50ipc')

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
    
    main(args)