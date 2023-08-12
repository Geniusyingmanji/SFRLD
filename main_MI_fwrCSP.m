clc; clearvars; close all; warning off all;
rng('default');
addpath(genpath('./utils/'));
%% Compare raw and aligned features on MI
%% Leave-one subject-out
%% need to enable covariancetoolbox
all=[];
mAcc=cell(1,2); mTime=cell(1,2);
for ds=1:1
    %% make data
% dataFolder=['../MI2-1/'];
% files=dir([dataFolder 'A*.mat']);

% dataFolder=['../MI2-6/'];
% files=dir([dataFolder 'F*.mat']);

% dataFolder=['../MI3T/'];%%文章用的
% files=dir([dataFolder 'B*.mat']);
% dataFolder=['../MI3/']; %%新
% files=dir([dataFolder 'B*.mat']);

dataFolder=['../MI5-1/'];
files=dir([dataFolder 'A*.mat']);
%     tic;
   % Ref=load([dataFolder 'Resting.mat']); % break time for all subjects
    XRaw=[]; yAll=[]; XAlignE=[]; XAlignR=[];
    XAlignLE=[];
    %%原始数据  标签      EA          RA
    for s=1:length(files)
        s
        load([dataFolder files(s).name]);
        x(isnan(x))=0;%%缺失值填充
        XRaw=cat(3,XRaw,x);
        yAll=cat(1,yAll,y); 
        nTrials=length(y);
     %   Bt=Ref.ref(:,:,(s-1)*nTrials+1:s*nTrials);
        RtE=mean(covariances(x),3); % reference state, Euclidean space
        RtR=riemann_mean(covariances(x)); % reference state, Riemmanian space
        Rtle=logeuclid_mean(covariances(x));

        sqrtRtE=RtE^(-1/2); %%相似性变换矩阵
        sqrtRtR=RtR^(-1/2);
        sqrtRtle=Rtle^(-1/2);
        
        XR=nan(size(x,1),size(x,2),nTrials);
        XE=nan(size(x,1),size(x,2),nTrials);
        LE=nan(size(x,1),size(x,2),nTrials);
        for j=1:nTrials
            XR(:,:,j)=sqrtRtR*x(:,:,j);
            XE(:,:,j)=sqrtRtE*x(:,:,j);
            LE(:,:,j)=sqrtRtle*x(:,:,j);
        end
        XAlignE=cat(3,XAlignE,XE); 
        XAlignR=cat(3,XAlignR,XR);
        XAlignLE=cat(3,XAlignLE,LE);
    end
    
    Accs=cell(1,length(files));
    times=cell(1,length(files));
    

    ALL_acc = [];
%     tic;
    for t=1:length(files)
        t
        yt=yAll((t-1)*nTrials+1:t*nTrials);
        ys=yAll([1:(t-1)*nTrials t*nTrials+1:end]);
        XtRaw=XRaw(:,:,(t-1)*nTrials+1:t*nTrials);%%目标域
        XsRaw=XRaw(:,:,[1:(t-1)*nTrials t*nTrials+1:end]);%%源域
        %%EA
        XtAlignE=XAlignE(:,:,(t-1)*nTrials+1:t*nTrials);%%目标域EA
        XsAlignE=XAlignE(:,:,[1:(t-1)*nTrials t*nTrials+1:end]);%%源域EA
        %%RA
        XtAlignR=XAlignR(:,:,(t-1)*nTrials+1:t*nTrials);%%目标域RA
        XsAlignR=XAlignR(:,:,[1:(t-1)*nTrials t*nTrials+1:end]);%%源域RA
        %%LE
        XtAlignLE=XAlignLE(:,:,(t-1)*nTrials+1:t*nTrials);%%目标域Le
        XsAlignLE=XAlignLE(:,:,[1:(t-1)*nTrials t*nTrials+1:end]);%%源域Le


    tic;%%计时器开始时间
    % Dataset 2a

%     [fTrain,fTest]=CSPfeature_s(XsAlignE, ys, XtAlignE, 8);%%MI3T时=3
%     options.T = 10;
%     options.dim = 14;
%     options.mu = 0.48;
%     options.lambda = 1.38;
%     addpath(genpath('./2021-LDA/'));
%     [acc,res]=LDA_DA_(fTrain',ys,fTest',yt,options,t);%无FWR 有LDA-DA
% 
%     LDA = fitcdiscr(fTrain,ys); %%无FWR 无LDA-DA
%     yPred=predict(LDA,fTest);
%     res=100*mean(yt==yPred);
    
   
%     res = Transfer_fwr(XsRaw, ys, XtRaw, yt,t);%%无EA
%     res = Transfer_fwr(XsAlignR, ys, XtAlignR, yt,t);%%RA
%     res = Transfer_fwr(XsAlignLE, ys, XtAlignLE, yt,t);%%logEA
    res = Transfer_fwr(XsAlignE, ys, XtAlignE, yt,t);%%EA

    % Dataset 2b
%     res = Transfer_fwr(XsAlignE, ys, XtAlignE, yt);
%      res = Transfer_fwr(XsAlignE, ys, XtAlignE, yt, 2, 2, 1.5, 0.1, 0.1);
    
    % Dataset 4a
%     res = Transfer_fwr(XsAlignE, ys, XtAlignE, yt, 18, 9, 9.5, 5, 5);
    

% res
    Accs{t}(1)=res(end);
    times{t}(1)=toc;
    end

%    res
    mAcc{ds}=[]; mTime{ds}=[];
    for t=1:length(files)
        mAcc{ds}=cat(1,mAcc{ds},Accs{t});
        mTime{ds}=cat(1,mTime{ds},times{t});
    end
    mAcc{ds}=cat(1,mAcc{ds},mean(mAcc{ds}));
    mTime{ds}=cat(1,mTime{ds},mean(mTime{ds}));
    mAcc{ds}

end
save('MIall.mat','mAcc','mTime');%%存储在“MIall.mat”文件中