%% 被试内模型--预处理+比较
clear;
clc;
% 获取数据
accuracy = load('result.mat');
data_data = accuracy.data;

data_data_1 = mean(data_data,3)';
kk = mean(data_data_1,1);

ITR = 60 * calculateITR( 40,0.6485,1.33 );
% 数据集选择
data_choose = 1;
if (data_choose == 1)
    Fs = 250;
    sum_class = 40;
    latencyDelay = round(0.14*Fs);
    subject = 35;
    num_of_subj = 35;
    ch_used=[48 54 55 56 57 58 61 62 63];
    num_of_subbands = 1;
    T = 2;
    Tw = 2;
    cross = 6;
    training_number = 5;
    total_training = sum_class * training_number;
    total_test = sum_class * 1;
    % Chebyshev Type I filter design
    for k=1:num_of_subbands
        Wp = [(8*k)/(Fs/2) 90/(Fs/2)];
        Ws = [(8*k-2)/(Fs/2) 100/(Fs/2)];
        [N,Wn] = cheb1ord(Wp,Ws,3,40);
        [subband_signal(k).bpB,subband_signal(k).bpA] = cheby1(N,0.5,Wn);
    end

    %notch
    Fo = 50;
    Q = 35;
    BW = (Fo/(Fs/2))/Q;

    [notchB,notchA] = iircomb(Fs/Fo,BW,'notch');
    seed = RandStream('mt19937ar','Seed','shuffle');

    % % 六折交叉验证的情况，初始化
    % EEG_subject_training_data = zeros(subject,total_training,cross,length(ch_used),floor(Tw * Fs));
    % EEG_subject_reference = zeros(subject,total_training,cross,sum_class,length(ch_used),floor(Tw * Fs));
    % EEG_subject_test_data = zeros(subject,total_test,cross,length(ch_used),floor(Tw * Fs));
    % EEG_training_label = zeros(subject,cross,total_training);
    % EEg_test_label = zeros(subject,cross,total_test);
    % EEG_subject_reference_test = zeros(subject,total_test,cross,sum_class,length(ch_used),floor(Tw * Fs));
    % 数据预处理和存储
    for sn = 1:num_of_subj
        str =  'D:\组会文献汇总\BCI-DATA\Benchmark Dataset\';
        str = strcat(str,'S',num2str(sn),'.mat\','S',num2str(sn));
        subject_data = load(str);
        data = subject_data.data;
        eeg=data(ch_used,floor(0.5*Fs)+1:floor(0.5*Fs+latencyDelay)+T *Fs,:,:);
        [d1_,d2_,d3_,d4_]=size(eeg);
        d1=d3_;d2=d4_;d3=d1_;d4=d2_;
        n_ch=d3;
        count = 0;
        for j=1:1:d2 % block数量
            for i=1:1:d1 %目标
                % for j=1:1:d2 % block数量
                count = count + 1;
                y0=reshape(eeg(:,:,i,j),d3,d4);
                y = filtfilt(notchB, notchA, y0.'); %notch
                y = y.';
                for sub_band=1:num_of_subbands
                    for ch_no=1:d3
                        tmp2=filtfilt(subband_signal(sub_band).bpB,subband_signal(sub_band).bpA,y(ch_no,:));
                        y_sb(ch_no,:) = tmp2(latencyDelay+1:latencyDelay+T*Fs);
                    end
                    EEG_data_total(sn,count,:,:) = y_sb(:,1:floor(Tw * Fs));
                    EEG_label(sn,count) = i;
                end
            end
        end
    end


else
    subject = 70;
    Fs = 250;
    latencyDelay = round(0.13*Fs);
    ch_used=[48 54 55 56 57 58 61 62 63];
    num_of_subbands = 1;
    T = 2;
    Tw = 0.6;
    % Chebyshev Type I filter design
    for k=1:num_of_subbands
        Wp = [(8*k)/(Fs/2) 90/(Fs/2)];
        Ws = [(8*k-2)/(Fs/2) 100/(Fs/2)];
        [N,Wn] = cheb1ord(Wp,Ws,3,40);
        [subband_signal(k).bpB,subband_signal(k).bpA] = cheby1(N,0.5,Wn);
    end
    %notch
    Fo = 50;
    Q = 35;
    BW = (Fo/(Fs/2))/Q;

    [notchB,notchA] = iircomb(Fs/Fo,BW,'notch');
    seed = RandStream('mt19937ar','Seed','shuffle');
    for sn = 1: subject
    str =  'E:\Matlab\matlab\OACCA\';
    str = strcat(str,'S',num2str(sn),'.mat');
    subject_data = load(str);
    data_1 = subject_data.data.EEG;
    px_1 = size(data_1,3);
    px_2 = size(data_1,4);
    for i = 1:px_1
        for j = 1:px_2
            data(:,:,j,i) = data_1(:,:,i,j);
        end
    end
    eeg=data(ch_used,floor(0.5*Fs)+1:floor(0.5*Fs+latencyDelay)+2*Fs,:,:);
    clear data;
    [d1_,d2_,d3_,d4_]=size(eeg);
    d1=d3_;d2=d4_;d3=d1_;d4=d2_;
    n_ch=d3;
    count = 0;
    for j=1:1:d2 % block数量
        for i=1:1:d1 %目标
            % for j=1:1:d2 % block数量
            count = count + 1;
            y0=reshape(eeg(:,:,i,j),d3,d4);
            y = filtfilt(notchB, notchA, y0.'); %notch
            y = y.';
            for sub_band=1:num_of_subbands
                for ch_no=1:d3
                    tmp2=filtfilt(subband_signal(sub_band).bpB,subband_signal(sub_band).bpA,y(ch_no,:));
                    y_sb(ch_no,:) = tmp2(latencyDelay+1:latencyDelay+T*Fs);
                end
                EEG_data_total(sn,count,:,:) = y_sb(:,1:floor(Tw * Fs));
                EEG_label(sn,count) = i;
            end
        end
    end
    end
end
value = 0;

%% 信息传输率函数
function ITR = calculateITR( N,P,T )

NMatrix = N*ones(size(P));

ITR = 1./T.*(log2(NMatrix)+P.*log2(P)+(1-P).*log2((1-P)./(N-1)));
ITR(P==0) = 1./T(P==0).*(log2(NMatrix(P==0))+(1-P(P==0)).*log2((1-P(P==0))./(N-1)));
ITR(P==1) = 1./T(P==1).*(log2(NMatrix(P==1))+P(P==1).*log2(P(P==1)));
end