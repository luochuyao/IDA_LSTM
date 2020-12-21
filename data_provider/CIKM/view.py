




import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from data_provider.CIKM.color_map import *
from scipy.misc import imread

def color_radar(img,flag=True):
    if flag:
        img = pixel_to_dBZ(img)
        img = mapping(img)
    else:
        pass
    return img

def pixel_to_dBZ(img):
    img = img.astype(np.float)/255.0
    img = img * 95.0
    img[img<15] = 0
    return img.astype(np.int)

RADAR_SAMPLE_INDEX = '2361'
def sample_radar_sequence(evaluate_folds,sample_index=RADAR_SAMPLE_INDEX):

    g_root = '/mnt/A/CIKM2017/CIKM_datasets/test/'
    p_root = '/mnt/A/meteorological/2500_ref_seq/'

    pre_res = []

    for evaluate_fold in evaluate_folds:

        sample_pred_path = p_root+evaluate_fold+'/sample_'+str(sample_index)+'/'
        model_res = []
        for i in range(6,16,1):
            img_path = sample_pred_path+'img_'+str(i)+'.png'
            pred_imgs = imread(img_path)
            pred_imgs = pred_imgs.astype(np.uint8)
            model_res.append(pred_imgs)

        pre_res.append(model_res)


    sample_ground_truth_path = g_root+'sample_'+str(sample_index)+'/'
    ground_truth_res = []
    for i in range(1,16,1):
        img_path = sample_ground_truth_path+'img_'+str(i)+'.png'
        real_img = imread(img_path)
        real_img = real_img.astype(np.uint8)
        ground_truth_res.append(real_img)
    return pre_res, ground_truth_res


def plot_radar(preds,ground_truths,root_path,img_name,flag = True):
    folds_num = len(preds)+1
    fig = plt.figure(figsize=(15,folds_num))
    gs = GridSpec(folds_num,15)
    # fig,ax = plt.subplots(nrows = 6,ncols = 15)
    for i in range(15):
        ax = plt.subplot2grid((folds_num,15),(0,i))
        ax.set_xticks([])
        ax.set_yticks([])
        if flag:
            ax.imshow(color_radar(ground_truths[i]))
        else:
            ax.imshow(ground_truths[i],cmap='Greys_r')

    for index in range(len(preds)):
        current_pred_imgs = preds[index]
        for i in range(10):
            ax = plt.subplot2grid((folds_num, 15), (index+1, i+5))
            if flag:
                ax.imshow(color_radar(current_pred_imgs[i]))
            else:
                ax.imshow(current_pred_imgs[i],cmap='Greys_r')
            ax.set_xticks([])
            ax.set_yticks([])
    print('save path is:'+str(img_name)+'view_radar.png')
    plt.savefig(os.path.join(root_path,str(img_name)+'view_radar.png'))



def inter_attn_w_ow_compare(flag=True):
    pred_root_path1 = '/mnt/A/meteorological/2500_ref_seq/CIKM_dst_predrnn'
    pred_root_path2 = '/mnt/A/meteorological/2500_ref_seq/CIKM_inter_dst_predrnn_r2'
    real_root_path = '/mnt/A/CIKM2017/CIKM_datasets/test/'
    sample_indexes = ['298','326','437','625','912','917','944','1985']
    sample_num = len(sample_indexes)
    real_imgs = []
    pred_img1s = []
    pred_img2s = []
    fig = plt.figure(figsize=(sample_num, 3))
    gs = GridSpec(3, sample_num)
    for i,index in enumerate(sample_indexes):
        real_fold = os.path.join(real_root_path,'sample_'+str(index))
        pred_fold1 = os.path.join(pred_root_path1,'sample_'+str(index))
        pred_fold2 = os.path.join(pred_root_path2,'sample_'+str(index))
        real_path = os.path.join(real_fold,'img_'+str(15)+'.png')
        pred_path1 = os.path.join(pred_fold1,'img_'+str(15)+'.png')
        pred_path2 = os.path.join(pred_fold2,'img_'+str(15)+'.png')
        real_img = imread(real_path)
        pred_img1 = imread(pred_path1)
        pred_img2 = imread(pred_path2)
        real_imgs.append(real_img)
        pred_img1s.append(pred_img1)
        pred_img2s.append(pred_img2)

        ax = plt.subplot2grid((3, sample_num), (0, i))
        ax.set_xticks([])
        ax.set_yticks([])
        if flag:
            ax.imshow(color_radar(real_img))
        else:
            ax.imshow(real_img, cmap='Greys_r')

        ax = plt.subplot2grid((3, sample_num), (1, i))
        ax.set_xticks([])
        ax.set_yticks([])
        if flag:
            ax.imshow(color_radar(pred_img1))
        else:
            ax.imshow(pred_img1, cmap='Greys_r')

        ax = plt.subplot2grid((3, sample_num), (2, i))
        ax.set_xticks([])
        ax.set_yticks([])
        if flag:
            ax.imshow(color_radar(pred_img2))
        else:
            ax.imshow(pred_img2, cmap='Greys_r')

    plt.savefig(os.path.join('/home/ices/', 'inter_view_w_wo.png'))
    print('inter_view_w_wo.png figrue has been generated ')

def attn_w_ow_compare(flag=True):
    pred_root_path1 = '/mnt/A/meteorological/2500_ref_seq/CIKM_sst_predrnn'
    pred_root_path2 = '/mnt/A/meteorological/2500_ref_seq/CIKM_cst_predrnn'
    pred_root_path3 = '/mnt/A/meteorological/2500_ref_seq/CIKM_dst_predrnn'
    real_root_path = '/mnt/A/CIKM2017/CIKM_datasets/test/'
    sample_indexes = ['909','923','932','2892','2900','2905','2906','2911']
    col_num = len(sample_indexes)
    real_imgs = []
    pred_img1s = []
    pred_img2s = []
    pred_img3s = []
    fig = plt.figure(figsize=(col_num, 4))
    gs = GridSpec(4, col_num)
    for i,index in enumerate(sample_indexes):
        real_fold = os.path.join(real_root_path,'sample_'+str(index))
        pred_fold1 = os.path.join(pred_root_path1,'sample_'+str(index))
        pred_fold2 = os.path.join(pred_root_path2,'sample_'+str(index))
        pred_fold3 = os.path.join(pred_root_path3,'sample_'+str(index))
        real_path = os.path.join(real_fold,'img_'+str(15)+'.png')
        pred_path1 = os.path.join(pred_fold1,'img_'+str(15)+'.png')
        pred_path2 = os.path.join(pred_fold2,'img_'+str(15)+'.png')
        pred_path3 = os.path.join(pred_fold3,'img_'+str(15)+'.png')
        real_img = imread(real_path)
        pred_img1 = imread(pred_path1)
        pred_img2 = imread(pred_path2)
        pred_img3 = imread(pred_path3)
        real_imgs.append(real_img)
        pred_img1s.append(pred_img1)
        pred_img2s.append(pred_img2)
        pred_img3s.append(pred_img3)

        ax = plt.subplot2grid((4, col_num), (0, i))
        ax.set_xticks([])
        ax.set_yticks([])
        if flag:
            ax.imshow(color_radar(real_img))
        else:
            ax.imshow(real_img, cmap='Greys_r')

        ax = plt.subplot2grid((4, col_num), (1, i))
        ax.set_xticks([])
        ax.set_yticks([])
        if flag:
            ax.imshow(color_radar(pred_img1))
        else:
            ax.imshow(pred_img1, cmap='Greys_r')

        ax = plt.subplot2grid((4, col_num), (2, i))
        ax.set_xticks([])
        ax.set_yticks([])
        if flag:
            ax.imshow(color_radar(pred_img2))
        else:
            ax.imshow(pred_img2, cmap='Greys_r')

        ax = plt.subplot2grid((4, col_num), (3, i))
        ax.set_xticks([])
        ax.set_yticks([])
        if flag:
            ax.imshow(color_radar(pred_img3))
        else:
            ax.imshow(pred_img2, cmap='Greys_r')

    plt.savefig(os.path.join('/home/ices/', 'attn_view_w_wo.png'))
    print('attn_view_w_wo.png figrue has been generated ')

def all_fig_save():
    # evaluate_folds = [
    #     "CIKM_convlstm",
    #     "CIKM_ConvGRU_test",
    #     "CIKM_TrajGRU_test",
    #     "CIKM_predrnn",
    #     "CIKM_predrnn_plus",
    #     "e3d_s_lstm_test_",
    #     "CIKM_MIM_test",
    #     "CIKM_dst_predrnn",
    #     "CIKM_inter_dst_predrnn_r2"
    # ]
    evaluate_folds = [
        # "CIKM_convlstm",
        # "CIKM_convlstm_test_r3",
        # "CIKM_predrnn",
        # "CIKM_predrnn_r4",
        # "CIKM_predrnn_plus",
        # "CIKM_predrnn_plus_r4",
        # "CIKM_dst_predrnn",
        # "CIKM_inter_dst_predrnn_r2"
    ]
    evaluate_folds = [
        "3d_predrnn_lstm_test",
        "e3d_s_lstm_test_",
        "e3d_c_lstm_test",
        "e3d_d_lstm_test",
    ]
    root_path = '/home/ices/Documents/atten_3d_predrnn_ablation_example1/'
    if not os.path.exists(root_path):
        os.mkdir(root_path)

    for i in range(1,4001):
        if i<2800:
            continue
        if i>3000:
            continue
        if i%200==0:
            print(i)
        pred, real = sample_radar_sequence(evaluate_folds, str(i))
        plot_radar(pred, real, root_path, str(i), True)

if __name__ == '__main__':
    inter_attn_w_ow_compare()