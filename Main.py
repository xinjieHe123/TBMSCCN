import sys
import os
from torch.utils.data import DataLoader
import  torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from scipy.io import loadmat,savemat
from torch.utils.data import Dataset
import hdf5storage
from Net_struct_2 import multi_ch_Corr,ContrastiveLoss,Conv_CA,SiamCA
import time
temperature = 0.17299
seed = 20230111
torch.manual_seed(seed) # CPU
torch.cuda.manual_seed(seed) # GPU
permutation = [42,49,50,51,52,53,56,57,58]
# params = {'tw': 275, 'Fs': 256, 'cl': 40, 'ch': len(permutation)}
dataset_choose = 1
model_choose = 1 # if model_choose == 1, proposed model else model_choose == 2, Bisaime 模型
slid_windows_choose = 2  # if slid_windows_choose == 1, 滑动时间窗方法 else model_choose == 2, 不使用这个方法
template_choose = 1

class my_dataset(Dataset):
    def __init__(self,data_tensor,template_tensor,label_tensor):
        assert data_tensor.size(0)==label_tensor.size(0)
        self.data_tensor=data_tensor
        self.template_tensor = template_tensor
        self.label_tensor=label_tensor

    def __getitem__(self, index):
        return self.data_tensor[index],self.template_tensor[index],self.label_tensor[index]

    def __len__(self):
        return self.data_tensor.size(0)

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

# 滑动时间窗数据增强函数
def slid_windows(data_length,training_data,training_label):
    # 滑动时间窗参数设置
    step = data_length
    sum_sample = training_data.shape[2]
    slid_count = (sum_sample - data_length) // step + 1
    tr_num = training_data.shape[0]
    new_training_data = []
    new_training_label = []
    for tr in range(tr_num):
        tr_data = training_data[tr,:,:]
        tr_label = training_label[tr]
        for sc in range(slid_count):
            # 获取滑动样本，start = 0 开始
            new_trail = tr_data[:,sc * step : sc * step + data_length]
            new_training_data.append(new_trail)
            new_training_label.append(tr_label)

    # 转化数据类型
    new_training_data = np.asarray(new_training_data)
    new_training_label = np.asarray(new_training_label)

    return new_training_data,new_training_label

test_accuracy = np.load('Accuracy_test_corr_LSTM.npy')
savemat('result.mat', {'data': test_accuracy})

if dataset_choose == 1: # Benchmark 数据集
    device = 'cuda'
    subNo = 35
    end_epoch = 800
    arti = 6
    target = 40

    test_accuracy = np.load('Accuracy_test_corr_LSTM.npy')
    savemat('result.mat', {'data': test_accuracy})

    # EEG_data = hdf5storage.loadmat('EEG_data.mat')
    eeg_data = hdf5storage.loadmat('slid_window_benchmark.mat')
    data = eeg_data['EEG_data_total']
    eeg_label = hdf5storage.loadmat('slid_window_benchmark_label.mat')
    label = eeg_label['EEG_label']

else:
    device = 'cuda'
    subNo = 100
    end_epoch = 800
    arti = 7
    target = 9

    str_path = 'D:\个人信息\个人论文\张量表征网络跨被试\code'
    # 读取数据
    dir_data = os.path.join(str_path, 'EEG_data_eldbeta.mat')
    eeg_data = hdf5storage.loadmat(dir_data)
    data = eeg_data['subject_eeg_data'] # 100,63,9,500
    # 读取标签
    dir_label = os.path.join(str_path, 'EEG_label_eldbeta.mat')
    eeg_label = hdf5storage.loadmat(dir_label)
    label = eeg_label['subject_label']



# 划分时间窗数据
tw_matrix = [1.0, 1.2]
Tw = len(tw_matrix)
Fs = 250

training_accuracy_sum = np.zeros([Tw, subNo, arti])
test_accuracy_sum = np.zeros([Tw, subNo, arti])

for tw in range(Tw):
    data_length = int(np.round(Fs * tw_matrix[tw]))
    params = {'tw': data_length, 'Fs': Fs, 'cl': target, 'ch': len(permutation)}
    for sub in range(subNo):
        # 获取第sub个被试下每个目标的训练数据和标签数据
        subject_data = data[sub, :, :, 0:data_length]
        subject_label = label[sub, :]
        for cross in range(arti):
            if dataset_choose == 1:
                index = np.arange(0, subject_data.shape[0], 1)
                test_index = np.arange(cross * target, cross * target + target)
                training_index = np.setdiff1d(index, test_index, True)
            else:
                index = np.arange(0, subject_data.shape[0], 1)
                test_index = index[cross::arti]
                training_index = np.setdiff1d(index, test_index, True)

            # 使用滑动时间窗
            if slid_windows_choose == 1:
                # 获取第一次交叉验证下的训练数据，测试数据以及标签和参考模板
                training_data_1 = data[sub,training_index, :, :]
                test_data = subject_data[test_index, :, :]
                # 保证标签必须为0开始
                # training_label = subject_label[training_index] - np.ones([len(training_index)])
                # test_label = subject_label[test_index] - np.ones([len(test_index)])
                training_label_1 = subject_label[training_index]
                test_label = subject_label[test_index]
                # 滑动时间窗函数
                training_data,training_label = slid_windows(data_length, training_data_1, training_label_1)
                # 保证标签必须为0开始
                training_label = training_label - np.ones([training_data.shape[0]])
                test_label = test_label - np.ones([len(test_index)])
                ##
                re_training_data = subject_data[training_index, :, :]
                re_training_label = subject_label[training_index] - np.ones([len(training_index)])

            else:
                training_data = subject_data[training_index, :, :]
                test_data = subject_data[test_index, :, :]
                # 保证标签必须为0开始
                training_label = subject_label[training_index] - np.ones([len(training_index)])
                test_label = subject_label[test_index] - np.ones([len(test_index)])

            if template_choose == 1:
                # 获取训练参考模板
                reference = np.zeros([target, training_data.shape[1], training_data.shape[2]])
                for tg in range(target):
                    index_index = np.where(training_label == tg)
                    target_eeg = training_data[index_index[0], :, :]
                    reference_1 = np.mean(target_eeg, 0)
                    reference[tg, :, :] = reference_1
                # 将平均模板值放大240倍
                reference = np.expand_dims(reference, axis=0)
            else:
                reference = np.zeros([target, re_training_data.shape[1], re_training_data.shape[2]])
                for tg in range(target):
                    index_index = np.where(re_training_label == tg)
                    target_eeg = re_training_data[index_index[0], :, :]
                    reference_1 = np.mean(target_eeg, 0)
                    reference[tg, :, :] = reference_1
                # 将平均模板值放大240倍
                reference = np.expand_dims(reference, axis=0)
            # 参考模板获取
            template = np.tile(reference, (training_data.shape[0], 1, 1, 1))
            template_test = np.tile(reference, (test_data.shape[0], 1, 1, 1))
            # 转化为tensor格式--训练数据
            training_data = torch.tensor(training_data, dtype=torch.float32)
            template = torch.tensor(template, dtype=torch.float32)
            training_label = torch.LongTensor(training_label)
            training_dataset = my_dataset(training_data, template, training_label)
            train_dataloader = DataLoader(training_dataset, batch_size=64, shuffle=True)

            # 转化为tensor格式--测试数据
            test_data = torch.tensor(test_data, dtype=torch.float32)
            template_test = torch.tensor(template_test, dtype=torch.float32)
            test_label = torch.LongTensor(test_label)
            test_dataset = my_dataset(test_data, template_test, test_label)

            # 初始化模型和损失
            if model_choose == 1:
                Net = Conv_CA(params).to(device)
            else:
                Net = SiamCA(params).to(device)
            # Net = Conv_CA(params).to(device)
            criterion = nn.CrossEntropyLoss()
            if torch.cuda.is_available():
                criterion = criterion.to(device)
            criterion_Contra = ContrastiveLoss(params=params).to(device)
            optimizer = optim.Adam(Net.parameters(), lr=0.003, betas=(0.9, 0.999))

            epoch = 0
            while True:
                start_time = time.time()
                Net.train()
                data_len = len(training_dataset)
                loss_len = len(train_dataloader)
                cl_losses = 0
                contrastive_losses = 0
                acc_train = 0
                acc_test = 0
                correct_num = 0
                # 每次batchsize的运行
                for x, y, z in train_dataloader:
                    # for i,batch in enumerate(train_dataloader):
                    # signal_1 = batch[0].to(device)
                    # reference_2 = batch[1].to(device)
                    # label_1 = batch[2].to(device)
                    signal_1 = x.to(device)
                    reference_2 = y.to(device)
                    label_1 = z.to(device)
                    if model_choose == 1:
                        output, corr = Net(signal_1, reference_2)
                    else:
                        output, corr = Net(signal_1, reference_2, params)
                    # output = torch.sigmoid(output)
                    # 计算中心损失和交叉熵损失
                    cl_loss = criterion(output, label_1)
                    cl_losses += cl_loss.item()
                    contrastive_loss = criterion_Contra(corr, label_1, temperature=temperature)
                    contrastive_losses += contrastive_loss

                    # 计算训练准确率
                    _, predicted = torch.max(output.data, 1)
                    correct = (predicted == label_1).sum().item()
                    correct_num += correct

                    # 反向传播
                    Net.zero_grad()
                    loss = cl_loss + contrastive_loss
                    loss.backward()
                    clip_gradient(optimizer, 3.)  # gradient clipping
                    optimizer.step()

                # 统计损失和训练准确率
                acc_train = (correct_num / data_len) * 100
                cl_losses = cl_losses / loss_len
                contrastive_losses = contrastive_losses / loss_len

                # 计算测试准确率
                x_test = test_dataset.data_tensor.to(device)
                y_test = test_dataset.template_tensor.to(device)
                z_test = test_dataset.label_tensor.to(device)
                if model_choose == 1:
                    output_test, corr_test = Net(x_test, y_test)
                else:
                    output_test, corr_test = Net(x_test, y_test, params)

                _, predict_test = torch.max(output_test.data, 1)
                correct_test = (predict_test == z_test).sum().item()
                total_test = z_test.size(0)
                accuracy_test = (float(correct_test) / total_test) * 100
                end_time = time.time()
                epoch_training_time = end_time - start_time
                print('time: {} - subject: {} - cross: {} '
                      'Epoch: {} - train_cl_loss: {:.4f} '
                      '- train_contrastive_loss: {:.4f} '
                      '- train_acc: {:.2f} - test_acc: {:.2f} '
                      '- epoch_training_time: {:.2f}'
                      .format(tw_matrix[tw], sub, cross,
                              epoch, cl_losses,
                              contrastive_losses,
                              acc_train, accuracy_test,
                              epoch_training_time))

                epoch += 1

                if epoch >= end_epoch:
                    break

            # 统计测试和验证的识别准确率
            training_accuracy_sum[tw, sub, cross] = acc_train
            test_accuracy_sum[tw, sub, cross] = accuracy_test

np.save('Accuracy_training_corr_LSTM', training_accuracy_sum)
np.save('Accuracy_test_corr_LSTM', test_accuracy_sum)
