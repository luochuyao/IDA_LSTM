
import os
import jpype
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from scipy.misc import imsave,imread
from skimage.measure import compare_ssim
evaluate_root = '/mnt/A/meteorological/2500_ref_seq/'
test_root = '/mnt/A/CIKM2017/CIKM_datasets/test/'

def dBZ_to_pixel(dBZ_img):
    return (dBZ_img.astype(np.float) + 10.0) * 255.0/ 95.0

def pixel_to_dBZ(data):
    dBZ = data.astype(np.float) * 95.0 / 255.0 - 10
    return dBZ

def get_hit_miss_counts_numba(prediction, truth, thresholds=None):
    """This function calculates the overall hits and misses for the prediction, which could be used
    to get the skill scores and threat scores:
    This function assumes the input, i.e, prediction and truth are 3-dim tensors, (timestep, row, col)
    and all inputs should be between 0~1
    Parameters
    ----------
    prediction : np.ndarray
        Shape: (seq_len, height, width)
    truth : np.ndarray
        Shape: (seq_len, height, width)
    mask : np.ndarray or None
        Shape: (seq_len, height, width)
        0 --> not use
        1 --> use
    thresholds : list or tuple
    Returns
    -------
    hits : np.ndarray
        (seq_len, len(thresholds))
        TP
    misses : np.ndarray
        (seq_len, len(thresholds))
        FN
    false_alarms : np.ndarray
        (seq_len, len(thresholds))
        FP
    correct_negatives : np.ndarray
        (seq_len, len(thresholds))
        TN
    """

    assert 3 == prediction.ndim
    assert 3 == truth.ndim
    assert prediction.shape == truth.shape


    ret = _get_hit_miss_counts_numba(prediction=prediction,
                                     truth=truth,
                                     thresholds=thresholds)
    return ret[:, :, 0], ret[:, :, 1], ret[:, :, 2], ret[:, :, 3]


def _get_hit_miss_counts_numba(prediction, truth, thresholds):
    seqlen, height, width = prediction.shape
    threshold_num = len(thresholds)
    ret = np.zeros(shape=(seqlen, threshold_num, 4), dtype=np.int32)

    for i in range(seqlen):
        for m in range(height):
            for n in range(width):
                for k in range(threshold_num):
                    bpred = prediction[i][m][n] >= thresholds[k]
                    btruth = truth[i][m][n] >= thresholds[k]
                    ind = (1 - btruth) * 2 + (1 - bpred)
                    ret[i][k][ind] += 1
                    # The above code is the same as:
                    # ret[i][j][k][0] += bpred * btruth
                    # ret[i][j][k][1] += (1 - bpred) * btruth
                    # ret[i][j][k][2] += bpred * (1 - btruth)
                    # ret[i][j][k][3] += (1 - bpred) * (1- btruth)
    return ret


class SeqHKOEvaluation(object):
    def __init__(self, seq_len, threholds=None):
        if threholds==None:
            self._thresholds = dBZ_to_pixel(np.array([5.0, 20.0, 40.0]))
        else:
            self._thresholds = threholds
        self._seq_len = seq_len
        self.begin()

    def begin(self):
        self._total_hits = np.zeros((self._seq_len, len(self._thresholds)), dtype=np.int)
        self._total_misses = np.zeros((self._seq_len, len(self._thresholds)), dtype=np.int)
        self._total_false_alarms = np.zeros((self._seq_len, len(self._thresholds)), dtype=np.int)
        self._total_correct_negatives = np.zeros((self._seq_len, len(self._thresholds)),
                                                 dtype=np.int)
        self._datetime_dict = {}


    def clear_all(self):
        self._total_hits[:] = 0
        self._total_misses[:] = 0
        self._total_false_alarms[:] = 0
        self._total_correct_negatives[:] = 0


    def update(self, gt, pred):
        """

        Parameters
        ----------
        gt : np.ndarray
        pred : np.ndarray

        Returns
        -------

        """

        assert gt.shape[0] == self._seq_len
        assert gt.shape == pred.shape


        # TODO Save all the mse, mae, gdl, hits, misses, false_alarms and correct_negatives
        hits, misses, false_alarms, correct_negatives = \
            get_hit_miss_counts_numba(prediction=pred, truth=gt,thresholds=self._thresholds)

        self._total_hits += hits
        self._total_misses += misses
        self._total_false_alarms += false_alarms
        self._total_correct_negatives += correct_negatives

    def calculate_stat(self):
        """The following measurements will be used to measure the score of the forecaster

        See Also
        [Weather and Forecasting 2010] Equitability Revisited: Why the "Equitable Threat Score" Is Not Equitable
        http://www.wxonline.info/topics/verif2.html

        We will denote
        (a b    (hits       false alarms
         c d) =  misses   correct negatives)

        We will report the
        POD = a / (a + c)
        FAR = b / (a + b)
        CSI = a / (a + b + c)
        Heidke Skill Score (HSS) = 2(ad - bc) / ((a+c) (c+d) + (a+b)(b+d))
        Gilbert Skill Score (GSS) = HSS / (2 - HSS), also known as the Equitable Threat Score
            HSS = 2 * GSS / (GSS + 1)

        Returns
        -------

        """

        a = self._total_hits.astype(np.float64)
        b = self._total_false_alarms.astype(np.float64)
        c = self._total_misses.astype(np.float64)
        d = self._total_correct_negatives.astype(np.float64)

        pod = a / (a + c)
        far = b / (a + b)
        csi = a / (a + b + c)
        n = a + b + c + d
        aref = (a + b) / n * (a + c)
        gss = (a - aref) / (a + b + c - aref)
        hss = 2 * gss / (gss + 1)

        # return pod, far, csi, hss, gss,
        return pod, far, csi, hss, gss

def start_jd():
    jarpath = "CIKM_Eva.jar"
    jvmPath = jpype.getDefaultJVMPath()
    jpype.startJVM(jvmPath, "-ea", "-Djava.class.path=%s" % (jarpath))
    javaClass = jpype.JClass("Main")
    jd = javaClass()
    return jd

def seq_eva_hss_csi_java(model_name,jd):

    result = jd.evaluate_seq(model_name,test_root,evaluate_root)
    hss = []
    csi = []

    for t in range(10):
        cur_hss = []
        cur_csi = []
        for i in range(3):
            cur_hss.append(result[0][t][i])
            cur_csi.append(result[1][t][i])

        hss.append(cur_hss)
        csi.append(cur_csi)
    hss = np.array(hss)
    csi = np.array(csi)

    return hss,csi

def plot_seq_hss_or_csi(datas,names,model_names,type):

    x = []
    for i in range(1, 11, 1):
        x.append(i * 6)
    threasholds = [5,20,40]
    for threashold_i in range(3):
        plt.figure(figsize=(7.5,5))
        for idx,name in enumerate(names):

            plt.plot(x,datas[name][:,threashold_i])


        plt.grid()
        plt.legend(model_names)
        plt.xticks(x)
        plt.xlabel('Leadtime (Minutes)')
        if type == 'HSS':
            plt.ylabel('Heidk Skill Score (HSS)')
        if type == 'CSI':
            plt.ylabel('Critical Success Index (CSI)')

        title = 'The ' + type + ' as threshold = ' + str(threasholds[threashold_i])
        plt.title(title)
        plt.savefig(type+'_'+str(threashold_i)+'.png')
        # plt.show()

def seq_eva_hss_csi(true_fold,pred_fold):
    hko_eva = SeqHKOEvaluation(10)
    valid_root_path = 'valid_test.txt'
    sample_indexes = np.loadtxt(valid_root_path)
    for index in sample_indexes:
        true_current_fold = os.path.join(true_fold,'sample_'+str(int(index)))
        pre_current_fold = os.path.join(pred_fold,'sample_'+str(int(index)))
        pred_imgs = []
        true_imgs = []
        for t in range(6, 16, 1):
            pred_path = os.path.join(pre_current_fold,'img_'+str(t)+'.png')
            true_path = os.path.join(true_current_fold,'img_'+str(t)+'.png')
            pred_img = imread(pred_path)
            true_img = imread(true_path)
            pred_imgs.append(pred_img)
            true_imgs.append(true_img)
        pred_imgs = np.array(pred_imgs).astype(np.float)
        true_imgs = np.array(true_imgs).astype(np.float)

        hko_eva.update(true_imgs,pred_imgs)

    pod, far, csi, hss, gss = hko_eva.calculate_stat()
    return hss,csi

def seq_hss_csi_test(test_model_list,model_names,is_java=True,is_plot=True):
    if is_java:
        jd = start_jd()
    test_model_hss = {}
    test_model_csi = {}
    for i,model in enumerate(test_model_list):
        if is_java:
            hss, csi = seq_eva_hss_csi_java(model, jd)
        else:
            hss,csi = seq_eva_hss_csi(test_root, os.path.join(evaluate_root, model))
        test_model_hss[model] = hss
        test_model_csi[model] = csi
        mean_hss = np.mean(hss, 0)
        mean_csi = np.mean(csi, 0)
        print('The hss and csi of "', model_names[i] ,'" is: ')
        print(hss.shape, mean_hss)
        print(csi.shape, mean_csi)
        print()

    if is_plot:
        plot_seq_hss_or_csi(test_model_hss,test_model_list,model_names,'HSS')
        plot_seq_hss_or_csi(test_model_csi,test_model_list,model_names,'CSI')

    if is_java:
        jpype.shutdownJVM()

    return test_model_hss,test_model_csi



def eval_test(true_fold,pred_fold,eval_type):
    res = 0
    # valid_root_path = '/home/ices/PycharmProject/IDAST_LSTM/data_provider/valid_test.txt'
    # with open(valid_root_path) as f:
    #     sample_indexes = f.read().split('\n')[:-1]
    sample_indexes = list(range(1,4001,1))
    for index in sample_indexes:

        true_current_fold = true_fold+'sample_'+str(index)+'/'
        pre_current_fold = pred_fold+'sample_'+str(index)+'/'
        pred_imgs = []
        true_imgs = []
        for t in range(6, 16, 1):
            pred_path = pre_current_fold+'img_'+str(t)+'.png'
            true_path = true_current_fold+'img_'+str(t)+'.png'
            pred_img = imread(pred_path)
            true_img = imread(true_path)
            pred_img = pred_img.astype(np.float32)
            true_img = true_img.astype(np.float32)
            pred_imgs.append(pred_img)
            true_imgs.append(true_img)
        pred_imgs = np.array(pred_imgs)
        true_imgs = np.array(true_imgs)
        # pred_imgs = pixel_to_dBZ(pred_imgs)
        # true_imgs = pixel_to_dBZ(true_imgs)

        pred_imgs = pred_imgs.astype(np.float)
        true_imgs = true_imgs.astype(np.float)


        if eval_type == 'mse':
            # sample_res = np.square(pred_imgs - true_imgs).mean()
            sample_res = np.mean(np.square(pred_imgs - true_imgs))
        elif eval_type == 'mae':
            # sample_res = np.abs(pred_imgs - true_imgs).mean()
            sample_res = np.mean(np.abs(pred_imgs - true_imgs))
        elif eval_type == 'ssim':
            sample_res = 0
            for t in range(10):
                ssim = compare_ssim(pred_imgs[t],true_imgs[t])
                sample_res = sample_res+ssim
            sample_res = sample_res/10.0
        elif eval_type == 'rmse':
            sample_res = np.sqrt(np.mean(np.square(pred_imgs - true_imgs)))

        res = res+sample_res
    res = res/len(sample_indexes)
    return res


def all_test():
    true_test_root = '/mnt/A/CIKM2017/CIKM_datasets/test/'
    # true_validation_root = '/mnt/A/CIKM2017/CIKM_datasets/validation/'
    pred_root = '/mnt/A/meteorological/2500_ref_seq/'
    test_model_list = [
        "CIKM_convlstm",
        "CIKM_ConvGRU_test",
        "CIKM_TrajGRU_test",
        "CIKM_predrnn",
        "CIKM_predrnn_plus",
        "e3d_s_lstm_test_",
        "CIKM_MIM_test",
        "CIKM_dst_predrnn",
        "CIKM_inter_dst_predrnn_r2"
    ]
    # test_model_list = [
    #     "CIKM_predrnn",
    #     "CIKM_sst_predrnn",
    #     "CIKM_cst_predrnn",
    #     "CIKM_dst_predrnn",
    # ]
    # test_model_list = [
    #         "CIKM_convlstm",
    #         "CIKM_convlstm_test_r1",
    #         "CIKM_convlstm_test_r2",
    #         "CIKM_convlstm_test_r3",
    #         "CIKM_convlstm_test_r4"
    # ]
    # test_model_list = [
    #     "CIKM_predrnn",
    #     "CIKM_predrnn_r1",
    #     "CIKM_predrnn_r2",
    #     "CIKM_predrnn_r3",
    #     "CIKM_predrnn_r4"
    # ]
    # test_model_list = [
    #     "CIKM_predrnn_plus",
    #     "CIKM_predrnn_plus_r1",
    #     "CIKM_predrnn_plus_r2",
    #     "CIKM_predrnn_plus_r3",
    #     "CIKM_predrnn_plus_r4"
    # ]
    # test_model_list = [
    #     "CIKM_dst_predrnn",
    #     "CIKM_inter_dst_predrnn_r1",
    #     "CIKM_inter_dst_predrnn_r2",
    #     "CIKM_inter_dst_predrnn_r3",
    #     "CIKM_inter_dst_predrnn_r4"
    # ]
    # test_model_mse = {}
    # for model in test_model_list:
    #     mse = eval_test(true_test_root, pred_root + model + '/', 'weight_mse')
    #     test_model_mse[model] = mse
    #     print('weight_mse model is:', model)
    #     print(test_model_mse[model])

    test_model_mse = {}
    for model in test_model_list:
        mse = eval_test(true_test_root, pred_root + model + '/', 'rmse')
        test_model_mse[model] = mse
        print('mse model is:', model)
        print(test_model_mse[model])

    # test_model_mae = {}
    # for model in test_model_list:
    #     mae = eval_test(true_test_root, pred_root + model + '/', 'mae')
    #     test_model_mae[model] = mae
    #     print('mae model is:', model)
    #     print(test_model_mae[model])
    #
    # print('ssim')
    # test_model_ssim = {}
    # for id, model in enumerate(test_model_list):
    #     ssim = eval_test(true_test_root, pred_root + model + '/', 'ssim')
    #     test_model_ssim[model] = ssim
    #     print('ssim model is:', model)
    #     print(test_model_ssim[model])
    # print('*' * 80)

def all_seq_test():
    test_model_list = [
        "CIKM_convlstm",
        "CIKM_ConvGRU_test",
        "CIKM_TrajGRU_test",
        "CIKM_predrnn",
        "CIKM_predrnn_plus",
        "e3d_s_lstm_test_",
        "CIKM_MIM_test",
        "CIKM_dst_predrnn",
        "CIKM_inter_dst_predrnn_r2"
    ]
    model_names = ['ConvLSTM',
                   'ConvGRU',
                   'TrajGRU',
                   'PredRNN',
                   'PredRNN++',
                   'E3D-LSTM',
                   'MIM',
                   'DA-LSTM',
                   'IDA-LSTM']

    # test_model_list = [
    #     "CIKM_predrnn",
    #     "CIKM_sst_predrnn",
    #     "CIKM_cst_predrnn",
    #     "CIKM_dst_predrnn",
    # ]
    # model_names = ['PredRNN',
    #                'SA-LSTM',
    #                'CA-LSTM',
    #                'DA-LSTM']
    test_model_list = [
    # "CIKM_convlstm",
    # "CIKM_convlstm_test_r4",
    # "CIKM_predrnn",
    # "CIKM_predrnn_r4",
    # "CIKM_predrnn_plus",
    # "CIKM_predrnn_plus_r4",
    # "CIKM_dst_predrnn",
    # "CIKM_inter_dst_predrnn_r2"
        ]
    model_names = [
        # 'ConvLSTM',
        # 'IConvLSTM',
        # 'PredRNN',
        # 'IPredRNN',
        # 'PredRNN++',
        # 'IPredRNN++',
        # 'DA-LSTM',
        # 'IDA-LSTM'
    ]
    seq_hss_csi_test(test_model_list, model_names, is_java=True, is_plot=True)




if __name__ == '__main__':
    all_test()