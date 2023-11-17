from mmaction.utils import register_all_modules
register_all_modules(init_default_scope=True)
import os
import sys
import os.path as osp
# sys.path.append("../")
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils
from tqdm import tqdm
from mmengine.config import Config, DictAction
from tools.utils import get_dataset, get_network, get_eval_pool, evaluate_synset, get_time, DiffAugment, ParamDiffAug, build_original_dataset
import wandb
import copy
import random
from mmaction.structures import ActionDataSample
from tools.reparam_module import ReparamModule
# from kmeans_pytorch import kmeans
from tools.cfg import CFG as cfg
import warnings
import yaml
from tools.networks import networks
from mmaction.registry import RUNNERS, MODELS
from tools.custom_dataset import SyntheticDataset
from tools.custom_aug import DiffAug

warnings.filterwarnings("ignore", category=DeprecationWarning)

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

def manual_seed(seed=0):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)

def main(args):

    manual_seed()

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(x) for x in args.device])

    if args.max_experts is not None and args.max_files is not None:
        args.total_experts = args.max_experts * args.max_files

    print("CUDNN STATUS: {}".format(torch.backends.cudnn.enabled))

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.skip_first_eva==False:
        eval_it_pool = np.arange(0, args.Iteration + 1, args.eval_it).tolist()
    else:
        eval_it_pool = np.arange(args.eval_it, args.Iteration + 1, args.eval_it).tolist()
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader, loader_train_dict, class_map, class_map_inv = get_dataset(args.dataset, args.data_path, args.batch_real, args.subset, args=args, im=args.im)
    # model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)
    model_eval_pool = ['alexnet']
    
    im_res = im_size[0]

    args.im_size = im_size

    accs_all_exps = dict() # record performances of all experiments
    for key in model_eval_pool:
        accs_all_exps[key] = []

    data_save = []

    if args.dsa:
        # args.epoch_eval_train = 1000
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
               project="VideoDataDistillation_DATM",
               config=args,
               )

    args = type('', (), {})()

    for key in wandb.config._items:
        setattr(args, key, wandb.config._items[key])

    args.dsa_param = dsa_params
    args.zca_trans = zca_trans

    if args.batch_syn is None:
        args.batch_syn = num_classes * args.ipc

    args.distributed = torch.cuda.device_count() > 1


    print('Hyper-parameters: \n', args.__dict__)
    # print('Evaluation model pool: ', model_eval_pool)

    
    ''' organize the real dataset '''
    images_all, indices_class = build_original_dataset(channel, num_classes, dst_train, class_map)
    
    def get_images(c, n): # get random n images from class c
        idx_shuffle = np.random.permutation(indices_class[c])[:n]
        return images_all[idx_shuffle]

    ''' initialize the synthetic data '''
    label_syn = torch.cat([torch.ones(args.ipc,dtype=torch.long)*i for i in range(num_classes)]) # [0,0,0, 1,1,1, ..., 9,9,9]
    image_syn = torch.randn(size=(num_classes * args.ipc, channel, im_size[0], im_size[1]), dtype=torch.float)
    syn_lr = torch.tensor(args.lr_teacher).to(args.device)
    
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
        # random.shuffle(expert_files)
        if args.max_files is not None:
            expert_files = expert_files[:args.max_files]

        expert_id = [i for i in range(len(expert_files))]
        random.shuffle(expert_id)

        print("loading file {}".format(expert_files[expert_id[file_idx]]))
        buffer = torch.load(expert_files[expert_id[file_idx]])
        if args.max_experts is not None:
            buffer = buffer[:args.max_experts]
        buffer_id = [i for i in range(len(buffer))]
        random.shuffle(buffer_id)

    if args.pix_init == 'real':
        print('initialize synthetic data from random real images')
        for c in range(num_classes):
            image_syn.data[c * args.ipc:(c + 1) * args.ipc] = get_images(c, args.ipc).detach().data
            
    
    elif args.pix_init == 'samples_predicted_correctly':
        if args.parall_eva==False:
            device = torch.device("cuda:0")
        else:
            device = args.device
        if cfg.Initialize_Label_With_Another_Model:
            Temp_net = get_network(args.Initialize_Label_Model, channel, num_classes, im_size, dist=False).to(device)  # get a random model
        else:
            Temp_net = get_network(args.model, channel, num_classes, im_size, dist=False).to(device)  # get a random model
        Temp_net.eval()
        Temp_net = ReparamModule(Temp_net)
        if args.distributed and args.parall_eva==True:
            Temp_net = torch.nn.DataParallel(Temp_net)
        Temp_net.eval()
        logits=[]
        label_expert_files = expert_files
        temp_params = torch.load(label_expert_files[0])[0][args.Label_Model_Timestamp]
        temp_params = torch.cat([p.data.to(device).reshape(-1) for p in temp_params], 0)
        if args.distributed and args.parall_eva==True:
            temp_params = temp_params.unsqueeze(0).expand(torch.cuda.device_count(), -1)
        for c in range(num_classes):
            data_for_class_c = get_images(c, len(indices_class[c])).detach().data
            n, _, w, h = data_for_class_c.shape
            selected_num = 0
            select_times = 0
            cur=0
            temp_img = None
            Wrong_Predicted_Img = None
            batch_size = 256
            index = []
            while len(index)<args.ipc:
                print(str(c)+'.'+str(select_times)+'.'+str(cur))
                current_data_batch = data_for_class_c[batch_size*select_times : batch_size*(select_times+1)].detach().to(device)
                if batch_size*select_times > len(data_for_class_c):
                    select_times = 0
                    cur+=1
                    temp_params = torch.load(label_expert_files[int(cur/10)%10])[cur%10][args.Label_Model_Timestamp]
                    temp_params = torch.cat([p.data.to(device).reshape(-1) for p in temp_params], 0).to(device)
                    if args.distributed and args.parall_eva==True:
                        temp_params = temp_params.unsqueeze(0).expand(torch.cuda.device_count(), -1)
                    continue
                logits = Temp_net(current_data_batch, flat_param=temp_params).detach()
                prediction_class = np.argmax(logits.cpu().data.numpy(), axis=-1)
                for i in range(len(prediction_class)):
                    if prediction_class[i]==c and len(index)<args.ipc:
                        index.append(batch_size*select_times+i)
                        index=list(set(index))
                select_times+=1
                if len(index) == args.ipc:
                    temp_img = torch.index_select(data_for_class_c, dim=0, index=torch.tensor(index))
                    break
            image_syn.data[c * args.ipc:(c + 1) * args.ipc] = temp_img.detach()
    else:
        print('initialize synthetic data from random noise')


    ''' training '''
    image_syn = image_syn.detach().to(args.device).requires_grad_(True)
    syn_lr = syn_lr.detach().to(args.device).requires_grad_(True)
    
    optimizer_img = torch.optim.SGD([image_syn], lr=args.lr_img, momentum=0.5)
    optimizer_lr = torch.optim.SGD([syn_lr], lr=args.lr_lr, momentum=0.5)


    
    optimizer_img.zero_grad()

    ###

    '''test'''
    def SoftCrossEntropy(inputs, target, reduction='average'):
        input_log_likelihood = -F.log_softmax(inputs, dim=1)
        target_log_likelihood = F.softmax(target, dim=1)
        batch = inputs.shape[0]
        loss = torch.sum(torch.mul(input_log_likelihood, target_log_likelihood)) / batch
        return loss

    # criterion = SoftCrossEntropy
    criterion = nn.CrossEntropyLoss()

    print('%s training begins'%get_time())

    '''------test------'''
    '''only sum correct predicted logits'''
    if args.pix_init == "samples_predicted_correctly":
        if cfg.Initialize_Label_With_Another_Model:
            Temp_net = get_network(args.Initialize_Label_Model, channel, num_classes, im_size, dist=False).to(device)  # get a random model
        else:
            Temp_net = get_network(args.model, channel, num_classes, im_size, dist=False).to(device)  # get a random model
        Temp_net.eval()
        Temp_net = ReparamModule(Temp_net)
        if args.distributed:
            Temp_net = torch.nn.DataParallel(Temp_net)
        Temp_net.eval()
        logits=[]
        batch_size = 256
        for i in range(len(label_expert_files)):
            Temp_Buffer = torch.load(label_expert_files[i])
            for j in Temp_Buffer:
                temp_logits = None
                for select_times in range((len(image_syn)+batch_size-1)//batch_size):
                    current_data_batch = image_syn[batch_size*select_times : batch_size*(select_times+1)].detach().to(device)
                    Temp_params = j[args.Label_Model_Timestamp]
                    Initialize_Labels_params = torch.cat([p.data.to(args.device).reshape(-1) for p in Temp_params], 0)
                    if args.distributed:
                        Initialize_Labels_params = Initialize_Labels_params.unsqueeze(0).expand(torch.cuda.device_count(), -1)
                    Initialized_Labels = Temp_net(current_data_batch, flat_param=Initialize_Labels_params)
                    if temp_logits == None:
                        temp_logits = Initialized_Labels.detach()
                    else:
                        temp_logits = torch.cat((temp_logits, Initialized_Labels.detach()),0)
                logits.append(temp_logits.detach().cpu())
        logits_tensor = torch.stack(logits)
        true_labels = label_syn.cpu()
        predicted_labels = torch.argmax(logits_tensor, dim=2).cpu()
        correct_predictions = predicted_labels == true_labels.view(1, -1)
        mask = correct_predictions.unsqueeze(2)
        correct_logits = logits_tensor * mask.float()
        correct_logits_per_model = correct_logits.sum(dim=0)
        num_correct_images_per_model = correct_predictions.sum(dim=0, dtype=torch.float)
        average_logits_per_image = correct_logits_per_model / num_correct_images_per_model.unsqueeze(1) 
        Initialized_Labels = average_logits_per_image

    elif args.pix_init == "real":
        ...
        # model_cfg = networks['alexnet']
        
        # Temp_net = MODELS.build(copy.deepcopy(model_cfg)).to(args.device)
        
        # Temp_net = ReparamModule(Temp_net)
        
        # if args.distributed:
        #     Temp_net = torch.nn.DataParallel(Temp_net)
        # Temp_net.eval()
        # Temp_params = buffer[0][-1]
        # Initialize_Labels_params = torch.cat([p.data.to(args.device).reshape(-1) for p in Temp_params], 0)
        
        # tensors = [t.unsqueeze(0) for t in image_syn]
        # labels = []
        # for l in label_syn:
        #     data_sample = ActionDataSample()
        #     data_sample.set_gt_label(l)
        #     labels.append(data_sample)
            
        # batch = {"inputs": tensors, "data_samples":labels}
        
        # if args.distributed:
        #     Initialize_Labels_params = Initialize_Labels_params.unsqueeze(0).expand(torch.cuda.device_count(), -1)
        #     data = Temp_net.module.module.data_preprocessor(batch, training=True)
        # else:
        #     data = Temp_net.module.data_preprocessor(batch, training=True)
        
        
        # Initialized_Labels = Temp_net(**data, flat_param=Initialize_Labels_params, mode='logit')

    # acc = np.sum(np.equal(np.argmax(Initialized_Labels.cpu().data.numpy(), axis=-1), label_syn.cpu().data.numpy()))
    # print('InitialAcc:{}'.format(acc/len(label_syn)))

    # label_syn = copy.deepcopy(Initialized_Labels.detach()).to(args.device).requires_grad_(True)
    # label_syn.requires_grad=True
    label_syn = label_syn.to(args.device)
    

    # optimizer_y = torch.optim.SGD([label_syn], lr=args.lr_y, momentum=args.Momentum_y)

    # del Temp_net

    best_val_scores = []
    
    for it in range(0, args.Iteration+1):
        save_this_it = False
        # wandb.log({"Progress": it}, step=it)
        ''' Evaluate synthetic data '''
        if it in eval_it_pool:
            for model_eval in model_eval_pool:
                print('-------------------------\nEvaluation\nmodel_train = %s, model_eval = %s, iteration = %d'%(args.model, model_eval, it))
                if args.dsa:
                    print('DSA augmentation strategy: \n', args.dsa_strategy)
                    print('DSA augmentation parameters: \n', args.dsa_param.__dict__)
                else:
                    print('DC augmentation parameters: \n', args.dc_aug_param)

                tmp_best_score = []
                for it_eval in range(args.num_eval):
                    # if args.parall_eva==False:
                    #     device = torch.device("cuda:0")
                    #     net_eval = get_network(model_eval, channel, num_classes, im_size, dist=False).to(device) # get a random model
                    # else:
                    #     device = args.device
                    #     net_eval = get_network(model_eval, channel, num_classes, im_size, dist=True).to(device) # get a random model

                    device = args.device
                    eval_labs = label_syn.detach().to(device)
                    with torch.no_grad():
                        image_save = image_syn.to(device)
                    
                    image_syn_eval, label_syn_eval = copy.deepcopy(image_save.detach()).to(device), copy.deepcopy(eval_labs.detach()).to(device) # avoid any unaware modification

                    # train_pipeline_cfg = [
                    #    dict(type='DiffAug', strategy='color_crop_cutout_flip_scale_rotate'), 
                    #    dict(type='PackActionInputs')
                    # ]
                    
                    train_pipeline_cfg = [
                                            dict(type='Flip', flip_ratio=0.5), 
                                            # dict(type='FormatShape', input_format='NCHW'),
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
                            pipeline=train_pipeline_cfg,)
                        )

                    cfg = Config.fromfile(args.config)

                    cfg.train_pipeline = train_pipeline_cfg
                    cfg.train_dataloader = train_dataloader
                    # merge cli arguments to config
                    cfg = merge_args(cfg, args)
                    
                    runner = RUNNERS.build(cfg)

                    # start training
                    runner.train()
                
                    best_val_score = runner.message_hub.get_info("best_score")
                    tmp_best_score.append(best_val_score)
                best_score = np.array(tmp_best_score)
                print("Best Val Score Mean:", best_score.mean(), "Std:", best_score.std())
                wandb.log({"Best Val Score Mean": best_score.mean(), "Best Val Score Std": best_score.std()})
                if len(best_val_scores)==0 or best_score.mean() > best_val_scores[-1]:
                    print("Save This Iter!")
                    save_this_it = True
                    best_val_scores.append(best_score.mean())
                
        if save_this_it or it % 1000 == 0:
            with torch.no_grad():
                image_save = image_syn.cuda()

                save_dir = os.path.join(".", "logged_files", args.dataset, wandb.run.name)

                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                image_save_name = "images_" if not save_this_it else "images_best_"
                label_save_name = "labels_" if not save_this_it else "labels_best_"
                lr_save_name = "lr_" if not save_this_it else "lr_best_"
                
                torch.save(image_save.cpu(), os.path.join(save_dir, image_save_name+"{}.pt".format(it)))
                torch.save(label_syn.cpu(), os.path.join(save_dir, label_save_name+"{}.pt".format(it)))
                torch.save(syn_lr.detach().cpu(), os.path.join(save_dir, lr_save_name+"{}.pt".format(it)))
                
                wandb.log({"Pixels": wandb.Histogram(torch.nan_to_num(image_syn.detach().cpu()))}, step=it)
                if save_this_it:
                    save_this_it = False
     
        wandb.log({"Synthetic_LR": syn_lr.detach().cpu()})

        model_cfg = networks['alexnet']

        student_net = MODELS.build(copy.deepcopy(model_cfg)).to(args.device)
        
        student_net = ReparamModule(student_net)

        if args.distributed:
            student_net = torch.nn.DataParallel(student_net)

        student_net.train()

        num_params = sum([np.prod(p.size()) for p in (student_net.parameters())])

        if args.load_all:
            expert_trajectory = buffer[np.random.randint(0, len(buffer))]
        else:
            expert_trajectory = buffer[buffer_id[expert_idx]]
            expert_idx += 1
            if expert_idx == len(buffer):
                expert_idx = 0
                file_idx += 1
                if file_idx == len(expert_files):
                    file_idx = 0
                    random.shuffle(expert_id)
                print("loading file {}".format(expert_files[expert_id[file_idx]]))
                if args.max_files != 1:
                    del buffer
                    buffer = torch.load(expert_files[expert_id[file_idx]])
                if args.max_experts is not None:
                    buffer = buffer[:args.max_experts]
                random.shuffle(buffer_id)

        # Only match easy traj. in the early stage
        if args.Sequential_Generation:
            Upper_Bound = args.current_max_start_epoch + int((args.max_start_epoch-args.current_max_start_epoch) * it/(args.expansion_end_epoch))
            Upper_Bound = min(Upper_Bound, args.max_start_epoch)
        else:
            Upper_Bound = args.max_start_epoch

        start_epoch = np.random.randint(args.min_start_epoch, Upper_Bound)

        starting_params = expert_trajectory[start_epoch]
        target_params = expert_trajectory[start_epoch+args.expert_epochs]
        target_params = torch.cat([p.data.to(args.device).reshape(-1) for p in target_params], 0)
        student_params = [torch.cat([p.data.to(args.device).reshape(-1) for p in starting_params], 0).requires_grad_(True)]
        starting_params = torch.cat([p.data.to(args.device).reshape(-1) for p in starting_params], 0)

        syn_images = image_syn
        y_hat = label_syn

        param_loss_list = []
        param_dist_list = []
        indices_chunks = []

        for step in range(args.syn_steps):
            if not indices_chunks:
                indices = torch.randperm(len(syn_images))
                indices_chunks = list(torch.split(indices, args.batch_syn))

            these_indices = indices_chunks.pop()

            x = syn_images[these_indices]
            this_y = y_hat[these_indices]


            if args.dsa and (not args.no_aug):
                x = DiffAugment(x, args.dsa_strategy, param=args.dsa_param)
            
            # print("X:", type(x),x)
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
                print("CE Loss:", ce_loss, "Grad:",grad)
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
        # optimizer_y.zero_grad()
        
        grand_loss.backward()

        if grand_loss<=args.threshold:
            # optimizer_y.step()
            optimizer_img.step()
            optimizer_lr.step()
        else:
            wandb.log({"falts": start_epoch}, step=it)



        wandb.log({"Grand_Loss": grand_loss.detach().cpu(),
                   "image_syn":image_syn.detach().cpu(),
                   "image_syn_grad": image_syn.grad,
                   })

        for _ in student_params:
            del _

        if it%10 == 0:
            print('%s iter = %04d, loss = %.4f' % (get_time(), it, grand_loss.item()))
        

    wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')
    
    parser.add_argument('config', help='train config file path')
    parser.add_argument("--cfg", type=str, default="")
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
    parser.add_argument('--im', type=int, default=224)

    args = parser.parse_args()
    
    cfg.merge_from_file(args.cfg)
    for key, value in cfg.items():
        arg_name = '--' + key
        parser.add_argument(arg_name, type=type(value), default=value)
    args = parser.parse_args()
    main(args)


