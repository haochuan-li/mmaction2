import torch
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


def show_dynamic(di, dir_path):
    di = di.cpu().numpy()
    di = di - np.min(di)
    di = 255 * di / np.max(di)
    di = np.uint8(di)
    im = Image.fromarray(di)
    # print(im.size)
    # print("Current dir:", os.getcwd())
    # save_dir = os.path.join(os.getcwd(), dir_path)
    # print("Save dir:", save_dir)
    # print("Save dir:", dir_path)
    os.makedirs(dir_path, exist_ok=True)
    print("Saving dynamic image at:", dir_path)
    im.save(os.path.join(save_dir,"di.jpeg"))
    # for i in di:
    #     cv2.imshow('image', i)
    #     cv2.waitKey(0)
    # cv2.imshow('image', di)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

def vl_nnarpool_temporal(X,ids,dzdy=None):
    forward = dzdy is None
    sz = X.size()
    # print("sz:",sz)

    if ids.numel() != sz[0]:
        raise ValueError('Error: ids dimension does not match with X!')

    nVideos = 1

    if forward:
        Y = torch.zeros([nVideos,sz[1],sz[2],sz[3]], dtype=X.dtype, device=X.device)
        # Y = torch.zeros_like(X)
    else:
        Y = torch.zeros(sz, dtype=X.dtype, device=X.device)
    
    # print("Y shape:", Y.shape)
    
    for v in range(nVideos):
        # pool among frames
        N = sz[0]
        # magic numbers
        fw = torch.zeros(N)
        if N == 1:
            fw = 1
        else:
            for i in range(1,N+1):
                fw[i-1] = torch.sum((2*(torch.arange(i, N+1, dtype=torch.float32)-N-1) / (torch.arange(i, N+1, dtype=torch.float32))))
        
        if forward:
            y = []
            for i,f in enumerate(fw):
                y.append(f*X[i])
        Y = torch.stack(y,dim=0).sum(dim=0)
        
    return Y

def compute_arp(images):
    if len(images) == 0:
        di = None
        return di

    if isinstance(images, list):
        imagesA = []
        for i in range(len(images)):
            if not isinstance(images[i], str):
                raise ValueError('images must be an array of images or cell of image names')
            img = cv2.imread(images[i])
            # print(img.shape)
            imagesA.append(torch.from_numpy(img))
        images = torch.stack(imagesA,dim=0)
    
        # images = images.permute(1,2,3,0)
    # print("video shape:",images.shape)
    N = images.shape[0]
    di = vl_nnarpool_temporal(images.float().cuda(), torch.ones(1,N))
    return di

if __name__ == "__main__":
    import os
    root = './data/ucf101/rawframes'
    
    # video class 
    subfolders = [f for f in os.listdir(root) if os.path.isdir(os.path.join(root, f))]
    
    for s in tqdm(subfolders, desc="Subfolders", unit="folders"):
        s_path = os.path.join(root, s)
        video_dir = os.listdir(s_path)
        for v in tqdm(video_dir, desc="Videos", unit="videos"):
            # "./data/ucf101/rawframes/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01"
            v_path = os.path.join(s_path, v)
            image_files = os.listdir(v_path)
            image_files = [os.path.join(v_path, f) for f in image_files]
            di = compute_arp(image_files)
            save_dir = v_path.replace("rawframes", "dynamicframes")
            # print("Save dir:", save_dir)
            show_dynamic(di, save_dir)