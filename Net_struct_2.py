import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F

# from Bi_siamle_1 import params

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class multi_ch_Corr(nn.Module):
    def __init__(self, params, num, **kwargs):
        self.tw = params['tw']
        self.Fs = params['Fs']
        self.cl = params['cl']
        self.num = num
        self.corr = None
        super(multi_ch_Corr, self).__init__(**kwargs)
    def forward(self, input, **kwargs):
        x = input[0]  # [bs, tw * kernel_size_2]  signal
        x_ = torch.reshape(x, (-1, self.tw, self.num, 1))  # [bs, tw, kernel_size_2, 1]

        t = input[1]  # [bs, 1, tw * kernel_size_2, cl] reference
        t_ = torch.reshape(t, (-1, self.tw, self.num, self.cl))  # [bs, tw, kernel_size_2, cl]

        corr_xt = torch.sum(x_*t_, dim=1)  # [bs, kernel_size_2, cl]
        corr_xx = torch.sum(x_*x_, dim=1)  # [bs, kernel_size_2, 1]
        corr_tt = torch.sum(t_*t_, dim=1)  # [bs, kernel_size_2, cl]
        self.corr = corr_xt/torch.sqrt(corr_tt)/torch.sqrt(corr_xx)  # [bs, kernel_size_2, cl]
        self.out = self.corr  # [bs, kernel_size_2, cl]
        self.out = torch.mean(self.out, dim=1)  # [bs, cl]
        return self.out

class ContrastiveLoss(nn.Module):
    def __init__(self, params):
        super(ContrastiveLoss, self).__init__()
    # note: please revise some parameter such as "num_classes" and "cl-1".
    def forward(self, input, label, temperature):  # input: [bs, cl]   label: [bs]
        pos_mask = F.one_hot(label, num_classes=40)  # [bs, cl]
        pos_sim = torch.sum(input * pos_mask, dim=1)  # [bs]
        pos_sim = torch.exp(pos_sim/temperature)  # [bs]

        neg_mask = (torch.ones_like(pos_mask) - pos_mask).bool()
        neg_sim = input.masked_select(neg_mask).view(-1, 40-1)  # [bs, cl-1]
        neg_sim = torch.exp(neg_sim/temperature)  # [bs, cl-1]
        neg_sim = torch.sum(neg_sim, dim=1)  # [bs]

        return (-torch.log(pos_sim / neg_sim)).mean()

# 所提方法模型改进：TBMSCCN
class Conv_CA(nn.Module):
    def __init__(self, params):
        super(Conv_CA, self).__init__()
        self.kernel_size_2 = 1
        # 训练数据卷积模块
        self.conv_block1 = nn.Sequential(nn.Conv2d(1, 16, kernel_size=(9,9), padding='same', bias=False),
                                         )
        self.conv_block2 = nn.Sequential(nn.Conv2d(16, 1, kernel_size=(1,9), padding='same', bias=False),
                                         )
        self.conv_block2_1 = nn.Sequential(nn.Conv2d(16, 1, kernel_size=(9, 1), padding='same', bias=False),
                                         )
        self.conv_block2_2 = nn.Sequential(nn.Conv2d(16, 1, kernel_size=(17, 1), padding='same', bias=False),
                                           )
        self.conv_block2_3 = nn.Sequential(nn.Conv2d(16, 1, kernel_size=(27, 1), padding='same', bias=False),
                                           )
        self.conv_block3 = nn.Sequential(nn.Conv2d(1, 1, kernel_size= (1,9), padding='valid', bias=False),
                                         # nn.Dropout2d(p = 0.75, inplace=False),
                                         )
        self.flatten1 = nn.Flatten()

        # 参考模板卷积
        self.conv_ref1 = nn.Sequential(nn.Conv2d(9, 40, kernel_size=(9,1), padding='same', bias=False),
        )
        self.conv_ref2 = nn.Sequential(nn.Conv2d(40,1, kernel_size=(9,1), padding='same', bias=False),
                                       # nn.Dropout2d(p = 0.15, inplace=False),
                                       )
        self.conv_ref2_1 = nn.Sequential(nn.Conv2d(40, 1, kernel_size=(9, 1), padding='same', bias=False),
                                       # nn.Dropout2d(p = 0.15, inplace=False),
                                       )
        self.conv_ref2_2 = nn.Sequential(nn.Conv2d(40, 1, kernel_size=(17, 1), padding='same', bias=False),
                                       # nn.Dropout2d(p = 0.15, inplace=False),
                                       )
        self.conv_ref2_3 = nn.Sequential(nn.Conv2d(40, 1, kernel_size=(27, 1), padding='same', bias=False),
                                       # nn.Dropout2d(p = 0.15, inplace=False),
                                       )

        self.corr = multi_ch_Corr(params=params, num=self.kernel_size_2)
        self.fc = nn.Linear(40,40)
        self.conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(5, 1), padding=(2, 0))
        # self.conv2 = nn.Conv2d(in_channels=9, out_channels=9, kernel_size=(1, 1), padding=(0, 0),groups=9)
        self.conv2 = nn.Conv2d(in_channels=40, out_channels=40, kernel_size=(1, 1), padding=(0, 0), groups=40)  # 数据集不同，参数就需要进行调整！

    def forward(self, input_1,input_2):
        sig = input_1.transpose(-1,-2)  # [bs,tw,ch]  = [100, 50, 9]
        ref = input_2.transpose(-1,-2)  # [bs, cl, tw, ch] = [100, 40, 50, 9]

        # 信号卷积
        sig = sig.unsqueeze(1)
        sig = self.conv_block1(sig)
        # sig = self.conv_block2(sig)
        sig_1 = self.conv_block2_1(sig)
        sig_2 = self.conv_block2_2(sig)
        sig_3 = self.conv_block2_3(sig)
        sig_4 = torch.cat([sig_1, sig_2], 1)
        sig = torch.cat([sig_4, sig_3], 1)
        sig = torch.mean(sig,dim=1)
        sig = sig.unsqueeze(1)
        sig = self.conv_block3(sig)
        sig = self.flatten1(sig)

        # 参考模板卷积,移位处理
        ref = torch.transpose(ref,1,3)
        ref = self.conv_ref1(ref)
        # ref = self.conv_ref2(ref)
        ref_1 = self.conv_ref2_1(ref)
        ref_2 = self.conv_ref2_2(ref)
        ref_3 = self.conv_ref2_3(ref)
        ref_4 = torch.cat([ref_1, ref_2], 1)
        ref = torch.cat([ref_3, ref_4], 1)
        ref = torch.mean(ref, dim=1)
        ref = ref.unsqueeze(1)


        # 相关性系数获取
        corr = self.corr([sig, ref])  # [bs, cl]
        out=corr.unsqueeze(2)
        out=out.unsqueeze(2) # [64,40,1,1]
        out=self.conv2(out)
        out=out.flatten(1)
        # 全连接层
        # contrastive_loss = self.contrastiveloss(corr, label)
        # out = torch.reshape(corr, [-1, params['cl'], 1, 1])  # [bs, cl, 1, 1]
        # out = torch.transpose(out, 1, 2)  # [bs, 1, cl, 1]
        # out = self.conv(out)  # [bs, 1, cl, 1]
        # out = torch.reshape(out, [-1, params['cl']])  # [bs, cl]
        # # out = self.fc(corr)

        return out,corr

# 对应SiamCA的方法
class SiamCA(nn.Module):
    def __init__(self, params):
        super(SiamCA, self).__init__()
        self.kernel_size_1 = 9
        self.kernel_size_2 = 9
        self.lstm1 = nn.LSTM(input_size=params['ch'], hidden_size=self.kernel_size_1, batch_first=True)  # out = 32*50*10
        self.lstm2 = nn.LSTM(input_size=self.kernel_size_1, hidden_size=self.kernel_size_2, batch_first=True)  # 32*50*20
        self.flatten1 = nn.Flatten()
        self.flatten2 = nn.Flatten()
        self.bn1 = nn.BatchNorm1d(self.kernel_size_2*params['tw'])
        self.bn2 = nn.BatchNorm1d(self.kernel_size_2*params['tw'])
        self.corr = multi_ch_Corr(params=params, num=self.kernel_size_2)
        # self.contrastiveloss = ContrastiveLoss(params=params)
        self.conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(5, 1), padding=(2, 0))
        # self.li = nn.Linear(in_features=)
        # self.conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 1), padding=(0, 0))

    def forward(self, input_1,input_2,params):
        sig = input_1.transpose(-1,-2)  # [bs,tw,ch]  = [100, 50, 9]
        ref = input_2.transpose(-1,-2)  # [bs, cl, tw, ch] = [100, 40, 50, 9]

        sig, _ = self.lstm1(sig)  # output size = [bs, tw, kernel_size_1]
        sig, _ = self.lstm2(sig)  # [bs, tw, kernel_size_2]
        sig = self.flatten1(sig)  # [bs, tw * kernel_size_2]
        sig = self.bn1(sig)  # [bs, tw * kernel_size_2]

        ref = torch.reshape(ref, (-1, params['tw'], params['ch']))  # [bs * cl, tw, ch]
        ref, _ = self.lstm1(ref)  # [bs*cl, tw, kernel_size_1]
        ref, _ = self.lstm2(ref)  # [bs*cl, tw, kernel_size_2]  # 两层参数需要共享
        ref = torch.reshape(ref, [-1, params['tw']*self.kernel_size_2])  # [bs * cl, tw * kernel_size_2]
        ref = self.bn2(ref)  # [bs * cl, tw * kernel_size_2]
        ref = torch.reshape(ref, [-1, params['cl'], params['tw'] * self.kernel_size_2, 1])  # [bs, cl, tw * kernel_size_2, 1]

        ref = torch.transpose(ref, 1, 3)  # [bs, 1, tw * kernel_size_2, cl]

        corr = self.corr([sig, ref])  # [bs, cl]

        # contrastive_loss = self.contrastiveloss(corr, label)
        out = torch.reshape(corr, [-1, params['cl'], 1, 1])  # [bs, cl, 1, 1]
        out = torch.transpose(out, 1, 2)  # [bs, 1, cl, 1]
        out = self.conv(out)  # [bs, 1, cl, 1]
        out = torch.reshape(out, [-1, params['cl']])  # [bs, cl]
        return out, corr

# 对应Conv_CA论文的方法
class Conv_CA_1(nn.Module):
    def __init__(self, params):
        super(Conv_CA_1, self).__init__()
        self.kernel_size_2 = 1
        # 训练数据卷积模块
        self.conv_block1 = nn.Sequential(nn.Conv2d(1, 16, kernel_size=(9,9), padding='same', bias=False),
                                         )
        self.conv_block2 = nn.Sequential(nn.Conv2d(16, 1, kernel_size=(1,9), padding='same', bias=False),
                                         )
        self.conv_block3 = nn.Sequential(nn.Conv2d(1, 1, kernel_size= (1,9), padding='valid', bias=False),
                                         # nn.Dropout2d(p = 0.75, inplace=False),
                                         )
        self.flatten1 = nn.Flatten()

        # 参考模板卷积
        self.conv_ref1 = nn.Sequential(nn.Conv2d(9, 40, kernel_size=(9,1), padding='same', bias=False),
        )
        self.conv_ref2 = nn.Sequential(nn.Conv2d(40,1, kernel_size=(9,1), padding='same', bias=False),
                                       # nn.Dropout2d(p = 0.15, inplace=False),
                                       )
        self.corr = multi_ch_Corr(params=params, num=self.kernel_size_2)
        self.fc = nn.Linear(40, 40)

    def forward(self, input_1,input_2,params):
        sig = input_1.transpose(-1,-2)  # [bs,tw,ch]  = [100, 50, 9]
        ref = input_2.transpose(-1,-2)  # [bs, cl, tw, ch] = [100, 40, 50, 9]

        # 信号卷积
        sig = sig.unsqueeze(1)
        sig = self.conv_block1(sig)
        sig = self.conv_block2(sig)
        sig = self.conv_block3(sig)
        sig = self.flatten1(sig)

        # 参考模板卷积,移位处理
        ref = torch.transpose(ref,1,3)
        ref = self.conv_ref1(ref)
        ref = self.conv_ref2(ref)


        # 相关性系数获取
        corr = self.corr([sig, ref])  # [bs, cl]
        # corr = self.fc(corr)
        return corr

