import os
import time
import torch
import shutil
import numpy as np
import matplotlib.pyplot as plt


def rm_data():
    dir_path = os.path.join(os.getcwd(), 'visual')

    if os.path.exists(dir_path):
        for filename in os.listdir(dir_path):
            file_path = os.path.join(dir_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')
    else:
        os.makedirs(dir_path)


def save_data(save_dict):
    save_dict = {key: value.detach().cpu().numpy() for key, value in save_dict.items()}
    file_path = f'visual/match_{str(time.time_ns())[5:-5]}.npy'
    np.save(file_path, save_dict, allow_pickle=True)


def save_img(save_dict):
    save_dict = {key: value.detach().cpu().numpy() for key, value in save_dict.items()}
    
    for frame_id in range(save_dict['src'].shape[0]): 
        plt.imshow(save_dict['src'][frame_id].squeeze(0))
        if save_dict.get('src_k', None) is not None:
            plt.scatter(save_dict['src_k'][0], save_dict['src_k'][1], s=0.1, c='red')
        plt.savefig(f'visual/scene_{frame_id}_src_{str(time.time_ns())[5:-5]}.png', dpi=300)
        plt.clf()
        plt.imshow(save_dict['tgt'][frame_id].squeeze(0))
        if save_dict.get('tgt_k', None) is not None:
            # for kpoints_t in save_dict['tgt_k'][frame_id]:
                # plt.scatter(kpoints_t[1], kpoints_t[0], s=0.1, c='red')
            plt.scatter(save_dict['tgt_k'][0], save_dict['tgt_k'][1], s=0.1, c='red')
        plt.savefig(f'visual/scene_{frame_id}_tgt_{str(time.time_ns())[5:-5]}.png', dpi=300)
        plt.clf()


def plot_correspondence(point0, point1, correspondence, feat0=None, feat1=None, shape=(160, 160), prefix='corr'): 
    height, width = shape
    if feat0 is not None and feat1 is not None:
        # 在两幅图像中间添加 2 像素的间隔
        combined_width = width * 2 + 2
        combined_image = np.ones((height, combined_width)) * np.max([feat0.max(), feat1.max()])
        
        # 将图像放置到组合图像中
        combined_image[:, :width] = feat0
        combined_image[:, width + 2:] = feat1
    
        # 绘制组合图像
        plt.figure(figsize=(12, 6))
        plt.imshow(combined_image, cmap='gray', interpolation='nearest')

     # 绘制对应线
    for corr in correspondence:
        x1, x2 = point0[corr[0]]
        y1, y2 = point1[corr[1]]
        plt.plot([y1, y2 + width + 2], [x1, x2], 'r-', lw=0.5, c='g')
    
    # 去掉白边
    plt.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    plt.savefig(f'visual/{prefix}_{str(time.time_ns())[5:-5]}.png', dpi=600)
    

def plot_point(point, H, W, other_point=None, assignment=None, prefix='point', scale = 4):
    import matplotlib.pyplot as plt
    if isinstance(H, list) or isinstance(W, tuple): 
        _, ax = plt.subplots(figsize=(((W[1]-W[0])/(H[1]-H[0]))*scale, scale))
    else:
        _, ax = plt.subplots(figsize=((W/H)*scale, scale))
    ax.scatter(point.detach().cpu()[:, 0], point.detach().cpu()[:, 1],
    # cmap='spectral',
    c='#7CBA59',
    s=8,
    linewidth=0,
    alpha=1,
    marker=".")
    if other_point is not None:
        ax.scatter(other_point.detach().cpu()[:, 0], other_point.detach().cpu()[:, 1],
        # cmap='spectral',
        c='#E98F36',
        s=8,
        linewidth=0,
        alpha=1,
        marker=".")
    
    if isinstance(H, list) or isinstance(W, tuple): 
        plt.xlim(W[0], W[1])
        plt.ylim(H[0], H[1])
    else:
        plt.xlim(0, W)
        plt.ylim(0, H)
    # for (center_x, center_y), r in zip(point.detach().cpu(), np.ones(point.size(0))*2):
    #     # 绘制圆
    #     circle = patches.Circle((center_x, center_y), r, edgecolor='red', facecolor='none', lw=0.2)
    #     ax.add_patch(circle)

    if assignment is not None:
        src_idx, tgt_idx = torch.where(assignment.detach().cpu() == 1.)
        for s_idx, t_idx in zip(src_idx, tgt_idx):
            plt.plot([point[s_idx, 0].item(), other_point[t_idx, 0].item()], 
             [point[s_idx, 1].item(), other_point[t_idx, 1].item()], 
             'red', linestyle='-', linewidth=0.5)

    plt.savefig(f'visual/{prefix}_{str(time.time_ns())[5:-5]}.png', dpi=800)