function yPred = classify_fwr(F_new, Ft_new, sw, m, ch, test, tr, tr2)
    F1 = F_new(:,1:tr);
    F2 = F_new(:,tr+1:end);

    F1 = F1.*repmat(sw',1,tr); F2 = F2.*repmat(sw',1,tr2);

    M1 = (mean(F1'))'; M2 = (mean(F2'))';
    C1 = cov(F1'); C2 = cov(F2'); C = C1+C2;
    [V, D] = eig(C); D = diag(D); 
    [D, Di] = sort(D, 'descend'); V = V(:,Di);

    a = D(1)*D(m)*(m-1)/(D(1)-D(m));
    b = (m*D(m)-D(1))/(D(1)-D(m));

    D(1:m) = D(1:m)+D(ch/2);%%%%%%%%%%%%%%%%%%%%%%
    D(m+1:ch) = a./((m+1:1:ch)+b)+D(ch/2);

    M1 = V'*M1; M2 = V'*M2;

    for i=1:test
        Ft_new(:,i) = Ft_new(:,i) .*sw';
        Ft_new(:,i) = V'*Ft_new(:,i);
        Dis = sum((Ft_new(:,i)-M1).^2./D);
        Dis = Dis - sum((Ft_new(:,i)-M2).^2./D);
        if Dis>0
            yPred(i) = 1;
        else
            yPred(i) = 0;
        end
    end
    
end