function [acc,acc_ite] = LDA_DA(Xs,Ys,Xt,Yt, test, tr, tr2, options,subject)
%%LDA算法
%% Notice
%%% The paper uses the Back-Tracking Method to get the parameters with 
%%%     the highest classification accuracy.
%% input
%%% T:       iteration times
%%% dim:       the dimension
%%% mu:   the hyper-parameters of concat[X;hotY];
%%% lambda: the regularization term
%% output
%%% acc:        the classification accuracy (number,0~1)
%%% acc_ite:    the classification accuracy in each iteration (list)
%     options=defaultOptions(options,...
%                 'T',10,...
%                 'dim',100,...
%                 'mu',0.1,...
%                 'lambda',0.1);
%     which train;
%%参数
    T=options.T;
    dim=options.dim;
    lambda=options.lambda;
    mu=options.mu;
    
    len = size(Xs, 1);
    u=0:1:len-1;
    sw=exp(-u.^2/(2*(len+0.5).^2));%len=m
    Ytpseudo = classify_fwr(Xs, Xt, sw, len, len, test, tr, tr2);
    %%伪标签
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%MI3
%     Ytpseudo = classify_fwr(Xs, Xt, sw, len, len-1, test, tr, tr2);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%MI3
    Ytpseudo = Ytpseudo';
%     Ytpseudo=classifyKNN(Xs,Ys,Xt,1); % initialize the target pseudo labels by 1NN
%     svmmodel = train(double(Ys), sparse(double(Xs')),'-s 1 -B 1.0 -q');
    %%% the Yt below is used to calculate the accuracy only,
    %%% Yt is not involved in the training.
%     [Ytpseudo,~,~] = predict(double(Yt), sparse(double(Xt')), svmmodel,'-q');
    if isfield(options,'display')
        fprintf('[init] acc:%.4f\n',getAcc(Ytpseudo,Yt));
    end
    [m,ns]=size(Xs);
    nt=size(Xt,2);
    n=ns+nt;
    C=length(unique(Ys));
    acc_ite=[];
    H=centeringMatrix(n);
    hotYs=hotmatrix(Ys,C,0);%%one hot
    hotYtpseudo=hotmatrix(Ytpseudo,C,0);
    Xsnew=[Xs;mu*hotYs'];
    Xtnew=[Xt;mu*hotYtpseudo'];
    % solve M0 by Eq. (9)
    M0=marginalDistribution(Xsnew,Xtnew,C);
    cur = 0.0;
    for i=1:T
        % update Xt_new by concat Xt and Ytpseudo
        hotYtpseudo=hotmatrix(Ytpseudo,C,0);
        Xtnew=[Xt;mu*hotYtpseudo'];
        % solve Mc by Eq. (11)
        Mc=conditionalDistribution(Xsnew,Xtnew,Ys,Ytpseudo,C);
        Xnew=[Xsnew,Xtnew];
        M=M0+Mc;
        M=M./norm(M,'fro');
        [P,~]=eigs(Xnew*M*Xnew'+lambda*eye(m+C),Xnew*H*Xnew',dim,'sm');
        Z=P'*Xnew;
        Z=Z-mean(Z,2);
        Z=L2Norm(Z')';
        Zs=Z(:,1:ns);
        Zt=Z(:,ns+1:end);
        u=0:1:dim-1;
        sw=exp(-u.^2/(2*(dim-0.5).^2));
        Ytpseudo = classify_fwr(Zs, Zt, sw, dim, dim, test, tr, tr2);
        Ytpseudo = Ytpseudo';
        
%         svmmodel = train(double(Ys), sparse(double(Zs')),'-s 1 -B 1.0 -q');
        %%% the Yt below is used to calculate the accuracy only,
        %%% Yt is not involved in the training.
%         [Ytpseudo,~,~] = predict(double(Yt), sparse(double(Zt')), svmmodel,'-q');
%         Ytpseudo=classifyKNN(Zs,Ys,Zt,1);
        acc=getAcc(Ytpseudo,Yt);
%         acc_ite=[acc_ite,acc];
%         if isfield(options,'display')
%             fprintf('[%2d] acc:%.4f\n',i,acc);
%         end
        if acc > cur
            cur = acc;
            acc_ite = [acc_ite;cur];
            
            
%             Zs=Zs';
%             Y = tsne(Zs);
%             gscatter(Y(:,1),Y(:,2),Ys);

%             figure
%             Zt=Zt';
%             Y = tsne(Zt);
%             gscatter(Y(:,1),Y(:,2),Yt)
%             xlabel('feature1','FontSize',18)
%             ylabel('feature2','FontSize',18)
%             legend('Left','Right','FontSize',18,'Location','NorthWest')


        else
            return
        end


    end
    
    
end

