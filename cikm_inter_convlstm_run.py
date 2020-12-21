import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse


from core.models.model_factory import Model
from core.utils import preprocess
from core.utils.util import *
import core.trainer as trainer
from data_provider.CIKM.data_iterator import *
import math
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='PyTorch video prediction model - Interaction ConvLSTM')

# training/test
parser.add_argument('--is_training', type=int, default=1)
# parser.add_argument('--device', type=str, default='gpu:0')
# data

parser.add_argument('--dataset_name', type=str, default='radar')
parser.add_argument('--is_parallel', type=bool, default=False)
parser.add_argument('--r', type=int, default=4)
parser.add_argument('--save_dir', type=str, default='checkpoints/cikm_convlstm_r4')
parser.add_argument('--gen_frm_dir', type=str, default='/mnt/A/meteorological/2500_ref_seq/CIKM_convlstm_test_r4/')
parser.add_argument('--input_length', type=int, default=5)
parser.add_argument('--total_length', type=int, default=15)
parser.add_argument('--img_width', type=int, default=128)
parser.add_argument('--img_channel', type=int, default=1)

# model
parser.add_argument('--model_name', type=str, default='interact_convlstm')
parser.add_argument('--pretrained_model', type=str, default='')
parser.add_argument('--num_hidden', type=str, default='64,64,64,64')
parser.add_argument('--filter_size', type=int, default=5)
parser.add_argument('--stride', type=int, default=1)
parser.add_argument('--patch_size', type=int, default=4)
parser.add_argument('--layer_norm', type=int, default=1)

# scheduled sampling
parser.add_argument('--scheduled_sampling', type=int, default=1)
parser.add_argument('--sampling_stop_iter', type=int, default=50000)
parser.add_argument('--sampling_start_value', type=float, default=1.0)
parser.add_argument('--sampling_changing_rate', type=float, default=0.00002)

# optimization
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--reverse_input', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--max_iterations', type=int, default=80000)
parser.add_argument('--display_interval', type=int, default=200)
parser.add_argument('--test_interval', type=int, default=2000)
parser.add_argument('--snapshot_interval', type=int, default=5000)
parser.add_argument('--num_save_samples', type=int, default=10)
parser.add_argument('--n_gpu', type=int, default=1)

args = parser.parse_args()
batch_size = args.batch_size

def padding_CIKM_data(frame_data):
    shape = frame_data.shape
    batch_size = shape[0]
    seq_length = shape[1]
    padding_frame_dat = np.zeros((batch_size,seq_length,args.img_width,args.img_width,args.img_channel))
    padding_frame_dat[:,:,13:-14,13:-14,:] = frame_data
    return padding_frame_dat

def unpadding_CIKM_data(padding_frame_dat):
    return padding_frame_dat[:,:,13:-14,13:-14,:]



def schedule_sampling(eta, itr):
    zeros = np.zeros((args.batch_size,
                      args.total_length - args.input_length - 1,
                      args.img_width // args.patch_size,
                      args.img_width // args.patch_size,
                      args.patch_size ** 2 * args.img_channel))
    if not args.scheduled_sampling:
        return 0.0, zeros

    if itr < args.sampling_stop_iter:
        eta -= args.sampling_changing_rate
    else:
        eta = 0.0
    random_flip = np.random.random_sample(
        (args.batch_size, args.total_length - args.input_length - 1))
    true_token = (random_flip < eta)
    ones = np.ones((args.img_width // args.patch_size,
                    args.img_width // args.patch_size,
                    args.patch_size ** 2 * args.img_channel))
    zeros = np.zeros((args.img_width // args.patch_size,
                      args.img_width // args.patch_size,
                      args.patch_size ** 2 * args.img_channel))
    real_input_flag = []
    for i in range(args.batch_size):
        for j in range(args.total_length - args.input_length - 1):
            if true_token[i, j]:
                real_input_flag.append(ones)
            else:
                real_input_flag.append(zeros)
    real_input_flag = np.array(real_input_flag)
    real_input_flag = np.reshape(real_input_flag,
                           (args.batch_size,
                            args.total_length - args.input_length - 1,
                            args.img_width // args.patch_size,
                            args.img_width // args.patch_size,
                            args.patch_size ** 2 * args.img_channel))
    return eta, real_input_flag

def wrapper_test(model):
    test_save_root = args.gen_frm_dir
    clean_fold(test_save_root)
    loss = 0
    count = 0
    index = 1
    flag = True
    img_mse, ssim = [], []

    for i in range(args.total_length - args.input_length):
        img_mse.append(0)
        ssim.append(0)

    real_input_flag = np.zeros(
        (args.batch_size,
         args.total_length - args.input_length - 1,
         args.img_width // args.patch_size,
         args.img_width // args.patch_size,
         args.patch_size ** 2 * args.img_channel))
    output_length = args.total_length - args.input_length
    while flag:
        dat, (index, b_cup) = sample(batch_size, data_type='test', index=index)
        dat = nor(dat)
        tars = dat[:, -output_length:]
        ims = padding_CIKM_data(dat)

        ims = preprocess.reshape_patch(ims, args.patch_size)
        img_gen, _ = model.test(ims, real_input_flag)

        img_gen = preprocess.reshape_patch_back(img_gen, args.patch_size)
        img_out = unpadding_CIKM_data(img_gen[:, -output_length:])

        mse = np.mean(np.square(tars - img_out))

        img_out = de_nor(img_out)
        loss = loss + mse
        count = count + 1

        bat_ind = 0
        for ind in range(index - batch_size, index, 1):
            save_fold = test_save_root + 'sample_' + str(ind) + '/'
            clean_fold(save_fold)
            for t in range(6, 16, 1):
                imsave(save_fold + 'img_' + str(t) + '.png', img_out[bat_ind, t - 6, :, :, 0])
            bat_ind = bat_ind + 1

        if b_cup == args.batch_size - 1:
            pass
        else:
            flag = False

    return loss / count


def wrapper_valid(model):
    loss = 0
    count = 0
    index = 1
    flag = True
    img_mse, ssim = [], []

    for i in range(args.total_length - args.input_length):
        img_mse.append(0)
        ssim.append(0)

    real_input_flag = np.zeros(
        (args.batch_size,
         args.total_length - args.input_length - 1,
         args.img_width // args.patch_size,
         args.img_width // args.patch_size,
         args.patch_size ** 2 * args.img_channel))
    output_length = args.total_length - args.input_length
    while flag:

        dat, (index, b_cup) = sample(batch_size, data_type='validation', index=index)
        dat = nor(dat)
        tars = dat[:, -output_length:]
        ims = padding_CIKM_data(dat)

        ims = preprocess.reshape_patch(ims, args.patch_size)
        img_gen, _ = model.test(ims, real_input_flag)
        img_gen = preprocess.reshape_patch_back(img_gen, args.patch_size)
        img_out = unpadding_CIKM_data(img_gen[:, -output_length:])


        mse = np.mean(np.square(tars-img_out))
        loss = loss+mse
        count = count+1
        if b_cup == args.batch_size-1:
            pass
        else:
            flag = False

    return loss/count




def wrapper_train(model):
    if args.pretrained_model:
        model.load(args.pretrained_model)


    eta = args.sampling_start_value
    best_mse = math.inf
    tolerate = 0
    limit = 2
    best_iter = None
    for itr in range(1, args.max_iterations + 1):

        ims = sample(
            batch_size=batch_size
        )
        ims = padding_CIKM_data(ims)

        ims = preprocess.reshape_patch(ims, args.patch_size)
        ims = nor(ims)
        eta, real_input_flag = schedule_sampling(eta, itr)

        cost = trainer.train(model, ims, real_input_flag, args, itr)


        if itr % args.display_interval == 0:
            print('itr: ' + str(itr))
            print('training loss: ' + str(cost))



        if itr % args.test_interval == 0:
            print('validation one ')
            valid_mse = wrapper_valid(model)
            print('validation mse is:',str(valid_mse))

            if valid_mse<best_mse:
                best_mse = valid_mse
                best_iter = itr
                tolerate = 0
                model.save()
            else:
                tolerate = tolerate+1

            if tolerate==limit:
                model.load()
                test_mse = wrapper_test(model)
                print('the best valid mse is:',str(best_mse))
                print('the test mse is ',str(test_mse))
                break


# if os.path.exists(args.save_dir):
#     shutil.rmtree(args.save_dir)
# os.makedirs(args.save_dir)
#
# if os.path.exists(args.gen_frm_dir):
#     shutil.rmtree(args.gen_frm_dir)
# os.makedirs(args.gen_frm_dir)
#
gpu_list = np.asarray(os.environ.get('CUDA_VISIBLE_DEVICES', '-1').split(','), dtype=np.int32)
args.n_gpu = len(gpu_list)
print('Initializing models')

model = Model(args)
model.load()
# test_mse = wrapper_test(model)
# print('test mse is:',str(test_mse))
# if args.is_training:
#     wrapper_train(model)
# else:
#     wrapper_test(model)