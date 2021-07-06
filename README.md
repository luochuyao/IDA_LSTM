
# IDA-LSTM

This is a Pytorch implementation of IDA-LSTM, a recurrent model for radar echo extrapolation (precipitation nowcasting) as described in the following paper:

A Novel LSTM Model with Interaction Dual Attention forRadar Echo Extrapolation, by Chuyao Luo, Xutao Li, Yongliang Wen, Yunming Ye, Xiaofeng Zhang.

# Setup

Required python libraries: torch (>=1.3.0) + opencv + numpy + scipy (== 1.0.0) + jpype1.
Tested in ubuntu + nvidia Titan with cuda (>=10.0).

# Datasets
We conduct experiments on CIKM AnalytiCup 2017 datasets: [CIKM_AnalytiCup_Address](https://tianchi.aliyun.com/competition/entrance/231596/information) or [CIKM_Rardar](https://drive.google.com/drive/folders/1IqQyI8hTtsBbrZRRht3Es9eES_S4Qv2Y?usp=sharing) 

# Training
Use any '.py' script to train these models. To train the proposed model on the radar, we can simply run the cikm_inter_dst_predrnn_run.py or cikm_dst_predrnn_run.py


You might want to change the parameter and setting, you can change the details of variable ‘args’ in each files for each model

The preprocess method and data root path can be modified in the data/data_iterator.py file

There are all trained models. You can download it following this address:[trained model](https://drive.google.com/file/d/1pnTSDoaKuKouu7y_j-QTq8dDBKVA-mPD/view)


# Evaluation
We give two approaches to evaluate our models. 


The first method is to check all predictions by running the java file in the path of CIKM_Eva/src (It is faster). You need to modify some information of path and make a .jar file to run

The second method is to run the evaluation.py in the path of data_provider/CIKM/

# Prediction samples
5 frames are predicted given the last 10 frames.

![Prediction vislazation](https://github.com/luochuyao/IDA_LSTM/blob/master/radar_res.png)

Besides, we also offer some prediction results of models including ConvGRU, TrajGRU, PredRNN++ and MIM ![Download Address](https://1drv.ms/u/s!AjADGxHd4nm8iGofrKsXGZIXkyre?e=h1DzmM) 


