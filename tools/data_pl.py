from mmaction.utils import register_all_modules

register_all_modules(init_default_scope=True)

import torch
from mmaction.registry import DATASETS

@TRANSFORMS.register_module()
class FrameDistill(BaseTransform):
    def __init__(self,
                 ipc: int = 50,
                 io_backend: str = 'disk',
                 decoding_backend: str = 'cv2',
                 **kwargs) -> None:
        self.ipc = ipc
        self.io_backend = io_backend
        self.decoding_backend = decoding_backend
        self.kwargs = kwargs
        self.file_client = None

    def build_original_dataset(self, dst_train, num_classes=101):
        class_map = {x:x for x in range(num_classes)}
        
        images_all = []
        labels_all = []
        indices_class = [[] for c in range(num_classes)]
        print("BUILDING DATASET")
        # dst_train is after augmentation
        for i in tqdm(range(len(dst_train))):
            sample = dst_train[i]
            inputs = sample['inputs']
            data_sample = sample['data_samples']
            # print("label:",data_sample.gt_label.item())
            # images_all.append(torch.unsqueeze(inputs, dim=0))
            images_all.append(inputs)
            labels_all.append(class_map[data_sample.gt_label.item()]) 

        print("Images:",type(images_all[0]),len(images_all))
        print("Labels:", type(labels_all[0]),len(labels_all))
        
        for i, lab in tqdm(enumerate(labels_all)):
            indices_class[lab].append(i)
        images_all = torch.cat(images_all, dim=0).to("cpu")
        labels_all = torch.tensor(labels_all, dtype=torch.long, device="cpu")
        
        print("Images_all Shape:", images_all.shape)
        print("Label_all Shape:", labels_all.shape)
        
        return images_all, indices_class
    
    def transform(self, results: dict) -> dict:
        im_size = [224,224]
        num_classes = 101
        channel = 3
        print("Result len:", len(results))
        images_all, indices_class = self.build_original_dataset(results)
        def get_images(c, n): # get random n images from class c
            idx_shuffle = np.random.permutation(indices_class[c])[:n]
            return images_all[idx_shuffle]
        # ipc = 70
        ''' initialize the synthetic data '''
        label_syn = torch.cat([torch.ones(self.ipc,dtype=torch.long)*i for i in range(num_classes)]) # [0,0,0, 1,1,1, ..., 9,9,9]
        # print("label_syn shape:", label_syn.shape)

        image_syn = torch.randn(size=(num_classes * self.ipc, channel, im_size[0], im_size[1]), dtype=torch.float)
        pix_init = 'real'
        if pix_init == 'real':
            print('initialize synthetic data from random real images')
            for c in range(num_classes):
                image_syn.data[c * self.ipc:(c + 1) * self.ipc] = get_images(c, self.ipc).detach().data
        else:
            print('initialize synthetic data from random noise')

        results['imgs'] = image_syn

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'io_backend={self.io_backend}, '
                    f'decoding_backend={self.decoding_backend})')
        return repr_str



train_pipeline_cfg = [
    dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=1),
    # dict(type='DecordDecode'),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(
        type='MultiScaleCrop',
        input_size=224,
        scales=(1, 0.875, 0.75, 0.66),
        random_crop=False,
        max_wh_scale_gap=1),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='PackActionInputs')
]

dataset_type = 'RawframeDataset'
data_root = './data/ucf101/rawframes'
data_root_val = './data/ucf101/rawframes'
split = 1  # official train/test splits. valid numbers: 1, 2, 3
ann_file_train = f'./data/ucf101/ucf101_train_split_{split}_rawframes.txt'
ann_file_val = f'./data/ucf101/ucf101_val_split_{split}_rawframes.txt'
ann_file_test = f'./data/ucf101/ucf101_val_split_{split}_rawframes.txt'

train_dataset_cfg=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=dict(img=data_root),
        pipeline=train_pipeline_cfg)

train_dataset = DATASETS.build(train_dataset_cfg)
print("TrainDataset:", len(train_dataset))

packed_results = train_dataset[0]

# print(torch.unsqueeze(packed_results[0],dim=0))
# print(torch.tensor(packed_results[1]).item())
print(packed_results)

inputs = packed_results['inputs']
data_sample = packed_results['data_samples']

print('shape of the inputs: ', inputs.shape, (inputs.unsqueeze(dim=0)).shape)

# 获取输入的信息
print(data_sample)
print('image_shape: ', data_sample.img_shape)
# print('num_clips: ', data_sample.num_clips)
# print('clip_len: ', data_sample.clip_len)

# # 获取输入的标签
print('label: ', data_sample.gt_label)

from mmengine.runner import Runner

# BATCH_SIZE = 2

train_dataloader_cfg = dict(
    batch_size=32,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=train_dataset)

train_data_loader = Runner.build_dataloader(dataloader=train_dataloader_cfg)

batched_packed_results = next(iter(train_data_loader))
print("Batch:", batched_packed_results)
batched_inputs = batched_packed_results['inputs']
batched_data_sample = batched_packed_results['data_samples']
# print("Batch:", batched_inputs.shape, batched_data_sample)
# assert len(batched_inputs) == BATCH_SIZE
# assert len(batched_data_sample) == BATCH_SIZE