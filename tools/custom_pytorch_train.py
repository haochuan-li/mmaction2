
from mmaction.utils import register_all_modules

register_all_modules(init_default_scope=True)
import torch
import torch.nn as nn
import numpy as np
import time, platform
import torch.optim as optim
from mmengine import track_iter_progress
from mmengine.runner import Runner
from torch.utils.data import Dataset, DataLoader, TensorDataset
from mmaction.registry import DATASETS, TRANSFORMS, MODELS, METRICS
from mmaction.structures import ActionDataSample
from tools.custom_train import parse_args
from tools.v_distill import build_original_dataset
from tools.utils import get_dataset
from mmengine.dataset.sampler import DefaultSampler
from mmengine.utils.dl_utils import (TORCH_VERSION, collect_env,
                                     set_multi_processing)
from mmengine.dist import (broadcast, get_dist_info, get_rank, get_world_size,
                           init_dist, is_distributed, master_only)

def custom_collate(batch):
    # print("Collate:", batch)
    tensors = [x[0].unsqueeze(0) for x in batch]
    labels = []
    for x in batch:
        data_sample = ActionDataSample()
        data_sample.set_gt_label(x[1])
        labels.append(data_sample)
    # labels = [x[1] for x in batch]
    
    batch = {"inputs": tensors, "gt_label": labels, "data_samples":labels}
    return batch

class SyntheticDataset(Dataset):
    def __init__(self, image_syn, label_syn):
        self.image_syn = image_syn
        self.label_syn = label_syn
        
    def __len__(self):
        return len(self.label_syn)
    
    def __getitem__(self, idx):
        
        return self.image_syn[idx], self.label_syn[idx]

def setup_env(env_cfg, distributed=False):
    """Setup environment.

        An example of ``env_cfg``::

            env_cfg = dict(
                cudnn_benchmark=True,
                mp_cfg=dict(
                    mp_start_method='fork',
                    opencv_num_threads=0
                ),
                dist_cfg=dict(backend='nccl', timeout=1800),
                resource_limit=4096
            )

        Args:
            env_cfg (dict): Config for setting environment.
        """
    if env_cfg.get('cudnn_benchmark'):
        torch.backends.cudnn.benchmark = True

    mp_cfg: dict = env_cfg.get('mp_cfg', {})
    set_multi_processing(**mp_cfg, distributed=distributed)

    # init distributed env first, since logger depends on the dist info.
    if distributed and not is_distributed():
        dist_cfg: dict = env_cfg.get('dist_cfg', {})
        init_dist('pytorch', **dist_cfg)

    _rank, _world_size = get_dist_info()

    timestamp = torch.tensor(time.time(), dtype=torch.float64)
    # broadcast timestamp from 0 process to other processes
    broadcast(timestamp)
    _timestamp = time.strftime('%Y%m%d_%H%M%S',
                                time.localtime(timestamp.item()))

    # https://github.com/pytorch/pytorch/issues/973
    # set resource limit
    if platform.system() != 'Windows':
        import resource
        rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
        base_soft_limit = rlimit[0]
        hard_limit = rlimit[1]
        soft_limit = min(
            max(env_cfg.get('resource_limit', 4096), base_soft_limit),
            hard_limit)
        resource.setrlimit(resource.RLIMIT_NOFILE,
                            (soft_limit, hard_limit))

def main():
    args = parse_args()
    args.device = 'cuda'

    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader, loader_train_dict, class_map, class_map_inv = get_dataset('ucf101')

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
    # label_syn = torch.tensor([np.ones(args.ipc,dtype=np.int_)*i for i in range(num_classes)], dtype=torch.long, requires_grad=False, device=args.device).view(-1) # [0,0,0, 1,1,1, ..., 9,9,9]

    # env_cfg = dict(
    # cudnn_benchmark=False,
    # mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    # dist_cfg=dict(backend='nccl'))
    
    # setup_env(env_cfg)

    

    # print(dataset[0])
    
    
    # tdl = Runner.build_dataloader(train_dataloader)
    # print(next(iter(train_dataloader)))
    # for i, (x, y) in enumerate(train_dataloader):
    #     print(i, x.shape, y.shape)
    #     break

    model_cfg = dict(
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
    model = MODELS.build(model_cfg)
    # print(model, sum(p.numel() for p in model.parameters() if p.requires_grad))
    
    # return

    image_syn_grad = image_syn.detach().to(args.device).requires_grad_(True)
    # print(image_save.requires_grad, image_save.device, id(image_save), image_syn_grad.requires_grad, image_syn_grad.device, id(image_syn_grad), image_syn.requires_grad, image_syn.device, id(image_syn))
    # syn_lr = syn_lr.detach().to(args.device).requires_grad_(True)
    optimizer_img = torch.optim.SGD([image_syn_grad], lr=0.1, momentum=0.5)
    # optimizer_lr = torch.optim.SGD([syn_lr], lr=args.lr_lr, momentum=0.5)
    optimizer_img.zero_grad()
    
    
    dataset = SyntheticDataset(image_syn,label_syn)
    print("Dataset:", dataset[0])
    train_dataloader = DataLoader(
                    batch_size=32,
                    num_workers=8,
                    collate_fn = custom_collate,
                    # shuffle = True,
                    persistent_workers=True,
                    sampler = DefaultSampler(dataset,shuffle=True),
                    # sampler=dict(type='DefaultSampler', shuffle=True),
                    dataset=dataset)

    metric_cfg = dict(type='AccMetric')
    
    metric = METRICS.build(metric_cfg)


    print("loader len:",len(train_dataloader))
    device = 'cuda' # or 'cpu'
    max_epochs = 2

    optimizer = optim.SGD(model.parameters(), lr=0.005,momentum=0.9,weight_decay=0.0001)

    for epoch in range(max_epochs):
        model.train()
        losses = []
        for data_batch in track_iter_progress((train_dataloader,len(train_dataloader))):
            # print("Raw Batch:", inputs, gt_label)
            # data_batch = dict(inputs=inputs, gt_label=gt_label, data_samples=None)
            # print("Batch:", data_batch)
            data = model.data_preprocessor(data_batch, training=True)
            # print("Input Data:", data['inputs'].shape)
            loss_dict = model(**data, mode='loss')
            loss = loss_dict['loss_cls']

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        print(f'Epoch[{epoch}]: loss ', sum(losses) / len(train_dataloader))
        with torch.no_grad():
            model.eval()
            for data_batch in track_iter_progress((train_dataloader,len(train_dataloader))):
                data = model.data_preprocessor(data_batch, training=False)
                predictions = model(**data, mode='predict')
                data_samples = [d.to_dict() for d in predictions]
                metric.process(data_batch, data_samples)

            acc = metric.acc = metric.compute_metrics(metric.results)
            for name, topk in acc.items():
                print(f'{name}: ', topk)
        # with torch.no_grad():
        #     model.eval()
        #     for data_batch in track_iter_progress(val_data_loader):
        #         data = model.data_preprocessor(data_batch, training=False)
        #         predictions = model(**data, mode='predict')
        #         data_samples = [d.to_dict() for d in predictions]
        #         metric.process(data_batch, data_samples)

        #     acc = metric.acc = metric.compute_metrics(metric.results)
        #     for name, topk in acc.items():
        #         print(f'{name}: ', topk)
    
    
    
    
main()





