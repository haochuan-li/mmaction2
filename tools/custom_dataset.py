from mmaction.utils import register_all_modules
register_all_modules(init_default_scope=True)

from mmengine.dataset import BaseDataset
from mmaction.registry import DATASETS

@DATASETS.register_module()
class SyntheticDataset(BaseDataset):
    def __init__(self, image_syn, label_syn, pipeline,
                 test_mode=False, 
                 **kwargs):
        self.image_syn = image_syn
        self.label_syn = label_syn
        super(SyntheticDataset, self).__init__(pipeline=pipeline, test_mode=test_mode, **kwargs)
        
    
    def load_data_list(self):
        data_list = []
        for x, y in zip(self.image_syn, self.label_syn):
            data_list.append(dict(imgs=x.unsqueeze(0).numpy(), label=y.numpy(), input_shape=x.unsqueeze(0).shape, img_shape=[224,224], modality="RGB"))
        # print("Before:",self.image_syn[0])
        # if self.requires_grad:
            # image_syn = image_syn.detach().requires_grad_(True)
        # print("After:", image_syn[0])
        # return [dict(inputs=x.unsqueeze(0), gt_label=y, data_samples=None) for x, y in zip(self.image_syn, self.label_syn)]
        
        return data_list

    def get_data_info(self, idx: int) -> dict:
        data_info = super().get_data_info(idx)
        
        return data_info