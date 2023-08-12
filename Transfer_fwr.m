function Acc = Transfer_fwr(xTrain, yTrain, xTest, yTest,subject)

% extract data for all epochs of the first class concatenated (EEG{1}) and 
% all epochs of the second class concatenated (EEG{2})
%m and sigma - parameter for regularization

% D1=EEG{1};
% D2=EEG{2};

xTrain = permute(xTrain, [2 1 3]);%%转置维度
xTest = permute(xTest, [2 1 3]);

cs=unique(yTrain);
D1=xTrain(:,:,yTrain==cs(1));%%xTrain中所有属于类别1的样本
D2=xTrain(:,:,yTrain==cs(2));%%xTrain中所有属于类别2的样本


[n, ch, tr] = size(D1); 
[n, ch, tr2] = size(D2);
C1 = zeros(ch,ch); C2 = C1;%%类别 1 和类别 2 的协方差矩阵

for i=1:tr
    Cr1 =  D1(:,:,i)'*D1(:,:,i);
    C1 = C1+Cr1./trace(Cr1);
end

for i=1:tr2
    Cr2 = D2(:,:,i)'*D2(:,:,i);
    C2 = C2+Cr2./trace(Cr2);
end

C1 = C1/tr; C2=C2/tr2;
%%LDA投影矩阵
[W, E] = eig(inv(C1+C2)*C1); 
E = diag(E);%%特征值矩阵
E = abs(E-0.5); [E, Ei] = sort(E, 'descend');
W = W(:,Ei);%%特征向量矩阵

for i=1:tr%%样本投影到子空间
    C1r(:,:,i) = W'*D1(:,:,i)'*D1(:,:,i)*W;
    F1(:,i) = log(diag(C1r(:,:,i)));
end
for i=1:tr2
    C2r(:,:,i) = W'*D2(:,:,i)'*D2(:,:,i)*W;
    F2(:,i) = log(diag(C2r(:,:,i)));
end


[n_t, ch_t, test] = size(xTest);
for i=1:test
    Ct(:,:,i) = W'*xTest(:,:,i)'*xTest(:,:,i)*W;
    Ft(:,i) = log(diag(Ct(:,:,i)));
end

F = cat(2, F1, F2);
%LDA-DA
options.T = 30;
% MI1、MI2时dim=2 4 6 8 10 ...24
% MI3时dim=2 4
% MI5时dim=2 4 6 8 10 ...118
options.dim = 10;
options.mu = 0.4;
options.lambda = 0.46;
[acc,Acc] = LDA_DA(F, yTrain, Ft, yTest, test, tr, tr2, options,subject);




 

% %PDALC
% options.T=5;
% options.t=15;
% options.dim=10;
% options.beta=0.06;
% options.lambda=0.32;
% options.mu=1;
% options.SRM_Kernel='linear';%SRM Kernel
% options.SRM_gamma=1;% SRM gamma
% options.SRM_mu=1;
% [acc,Acc] = PDALC(F, yTrain, Ft, yTest, test, tr, tr2, options);


%DGB_DA
% options.T=10;
% options.subsetRate=0.8;
% options.p=15;
% options.alpha=0.9;
% options.delta=0.2;
% options.dim=10;
% [acc,Acc] = DGB_DA(F, yTrain, Ft, yTest, test, tr, tr2, options);

%CMFC
% options.alpha=0.1;
% options.lambda=0.01;
% options.gamma=30;
% options.beta=0.5;
% options.dim=100;
% options.T=10;
% options.kernel_type='primal';
% options.kernel_param=1;
% addpath(genpath('./2021-CMFC/'));
% CMFC(F, yTrain, Ft, yTest,test, tr, tr2, options);

% [acc,Acc] = PDALC(F, yTrain, Ft, yTest, test, tr, tr2, options);
% 
% T = 10;
% options.dim = p;
% options.lambda = beta;
% options.kernel_type = 'rbf';
% options.mu = alpha;
% options.gamma = 1.0;
% Cls = []; Acc = [];
% cur = 0.0;
% for t = 1:T  
%     [F_new,Ft_new] = JPDA(F, Ft, yTrain, Cls, options);
%     Cls = classify_fwr(F_new, Ft_new, sw, m, p, test, tr, tr2);
%     acc = length(find(Cls'==yTest))/length(yTest);
%     if acc > cur
%         cur = acc;
%         Acc = [Acc;cur];
%     else
%         return
%     end
% end










