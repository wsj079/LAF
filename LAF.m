clear
clc
addpath library
dataset = 'APY';%select dataset
distance = 'cosine';%set distance type
%% load data
if(strcmp(dataset, 'AWA1'))
    load ..\RSAE\xlsa17\data\AWA1\att_splits.mat
    load ..\RSAE\xlsa17\data\AWA1\res101.mat
    Ytr = labels(trainval_loc);
    Xtr = features(:,trainval_loc);
    Xtr = NormalizeFea(Xtr')';
    Xts = features(:,test_seen_loc);  Xts = NormalizeFea(Xts')';
    Xtu = features(:,test_unseen_loc);  Xtu = NormalizeFea(Xtu')';
elseif(strcmp(dataset, 'AWA2'))
    load ..\RSAE\xlsa17\data\AWA2\att_splits.mat
    load ..\RSAE\xlsa17\data\AWA2\res101.mat
    Ytr = labels(trainval_loc);
    Xtr = features(:,trainval_loc);  Xtr = NormalizeFea(Xtr')';
    Xts = features(:,test_seen_loc);  Xts = NormalizeFea(Xts')';
    Xtu = features(:,test_unseen_loc);  Xtu = NormalizeFea(Xtu')';
elseif(strcmp(dataset, 'CUB'))
    load ..\RSAE\xlsa17\data\CUB\att_splits.mat
    load ..\RSAE\xlsa17\data\CUB\res101.mat
    Ytr = labels(trainval_loc);
    Xtr = features(:,trainval_loc);
    Xtr = NormalizeFea(Xtr')';
    Xts = features(:,test_seen_loc);  Xts = NormalizeFea(Xts')';
    Xtu = features(:,test_unseen_loc);  Xtu = NormalizeFea(Xtu')';
elseif(strcmp(dataset, 'SUN'))
    load ..\RSAE\xlsa17\data\SUN\att_splits.mat
    load ..\RSAE\xlsa17\data\SUN\res101.mat
    Ytr = labels(trainval_loc);
    Xtr = features(:,trainval_loc);  Xtr = NormalizeFea(Xtr')';
    Xts = features(:,test_seen_loc);  Xts = NormalizeFea(Xts')';
    Xtu = features(:,test_unseen_loc);  Xtu = NormalizeFea(Xtu')';
elseif(strcmp(dataset, 'APY'))
    load ..\RSAE\xlsa17\data\APY\att_splits.mat
    load ..\RSAE\xlsa17\data\APY\res101.mat
    Ytr = labels(trainval_loc);
    Xtr = features(:,trainval_loc);  Xtr = NormalizeFea(Xtr')';
    Xts = features(:,test_seen_loc);  Xts = NormalizeFea(Xts')';
    Xtu = features(:,test_unseen_loc);  Xtu = NormalizeFea(Xtu')';
end
Yts = labels(test_seen_loc);
Ytu = labels(test_unseen_loc);
tr_ind = unique(Ytr,'stable');
ts_ind = unique(Yts,'stable');
tu_ind = unique(Ytu,'stable');
tu_attr = att(:,tu_ind);

Ax = zeros([size(att,1) size(Xtr,1)]);
for ii = 1:length(tr_ind)
    ind = find(Ytr==tr_ind(ii));
    Ax(:,ind) = repmat( att(:,tr_ind(ii)), 1, length(ind));
end

Y = zeros(length(tr_ind), size(Xtr,2));
for ii = 1:length(tr_ind)
    ind = find(Ytr==tr_ind(ii));
    Y(ii,ind) = 1;
end
%%%%% Training
numIter = 2000;

countRst = 0;
for alpha = 1
    for lambda = 0.01
        for beta = 0
            for miu = 0.001
              %% =====Initialize parameters========================
                % nfea：原始空间样本维度，nsamp：样本数
                [nfea,nsamp] = size(Xtr);
                % nsemantic：语义空间样本维度
                nsemantic = size(Ax,1);
                nclass = size(Y,1);
                % 最大的miu值
                miuMax = 1e+6;
                % 参数rho，一般取1.1
                rho = 1.1;
                % 约束条件无穷范数的阈值
                yita = 1e-3;
                % 指定初始化方式
                initWays = 'zeros';
                if strcmp(initWays,'zeros')
                    P = zeros(nclass,nfea);
                    Q = zeros(nsemantic,nclass);
                    Z = zeros(nclass,nsamp);
                    E = zeros(nclass,nsamp);
                    K1 = zeros(nclass,nsamp);
                    K2 = zeros(nclass,nsamp);
                    K3 = zeros(nsemantic,nsamp);
                elseif strcmp(initWays,'rand')
                    P = rand(nclass,nfea);
                    Q = rand(nsemantic,nclass);
                    Z = rand(nclass,nsamp);
                    E = rand(nclass,nsamp);
                    K1 = rand(nclass,nsamp);
                    K2 = rand(nclass,nsamp);
                    K3 = rand(nsemantic,nsamp);
                end
                % 迭代次数
                k = 1;
                % 3个约束条件的无穷范数
                fun1 = 1;
                fun2 = 1;
                fun3 = 1;
                % 3个约束条件的无穷范数数组，每个元素代表一次迭代结果
                fun1All = [];
                fun2All = [];
                fun3All = [];
                % 是否写入文件
                isWrite = true;
                % 迭代循环体，fun1/2/3都小于yita时，或迭代达到最大次数时，停止循环
                if isWrite
                    fid = fopen('output/fuxian_test.txt','a+');
                    fprintf(fid,'alpha = %f, lambda = %f, beta = %f, miu = %f\n', alpha,lambda,beta,miu);
                end
                P = (Xtr * Xtr' + alpha * eye(nfea))\Xtr*(Y');
                while  (fun1>=yita || fun2>=yita || fun3>=yita) && k<=numIter
                    countRst = countRst+1;
                    disp(num2str(k))
                    tic
                    
                    % update Z
                    M = 0.5 * (Y - E + Q' * Ax) + (K1 - K2) / (2 * miu);
                    [U,S_svd,V] = svd(M,'econ');
                    miuNew = 1 / (2 * miu);
                    [C] = threshold(S_svd,miuNew);% 求S(x)
                    Z = U * C * V';
                    
                    % update E
                    temp = Y - Z + K1 / miu;
                    tau = lambda / miu;
                    [E] = threshold(temp,tau);
                    
                    % update Q
                    S1 = miu * Ax * Ax' + beta*eye(nsemantic);
                    S2 = miu * Y * Y';
                    S3 = miu * (Ax*Z'+Ax*Y') + Ax*K2' + K3*Y';
                    Q = sylvester(S1,S2,S3);
                    
                    % update Y1,Y2 and Y3
                    K1 = K1 + miu * (Y - Z - E);
                    K2 = K2 + miu * (Z - Q' * Ax);
                    K3 = K3 + miu * (Ax - Q * Y);
                    
                    % update miu
                    miu = min(rho * miu, miuMax);
                    
                    % check convergence
                    fun1 = norm(Y - Z - E,'inf');
                    fun2 = norm(Z - Q' * Ax,'inf');
                    fun3 = norm(Ax - Q * Y,'inf');
                    fun1All = [fun1All fun1];
                    fun2All = [fun2All fun2];
                    fun3All = [fun3All fun3];
                    
                    % classification
                    X_te_pro =  tu_attr' *Q;
                    dist = pdist2(Xtu'*P, X_te_pro, distance);
                    [~, predict_label] = min(dist, [], 2);
                    zsl_unseen_predict_label = mapLabel(predict_label, tu_ind);
                    zsl = computeAcc(zsl_unseen_predict_label, Ytu, tu_ind)*100;
                    fprintf('[1] %s ZSL accuracy [X >>> Y <<< A]: %.1f%%\n', dataset, zsl);
                    
                    X_te_pro = att' *Q;
                    for ii = 1:length(tr_ind)
                        X_te_pro(tr_ind(ii),:) = 0;
                        X_te_pro(tr_ind(ii),ii) = 1;
                    end
                    dist = pdist2(Xtu'*P, X_te_pro, distance);
                    [~, predict_label] = min(dist, [], 2);
                    gzslu = computeAcc(predict_label, Ytu, tu_ind)*100;
                    fprintf('[2] %s GZSL unseen->all accuracy [X >>> Y <<< A]: %.1f%%\n', dataset, gzslu);
                    
                    dist = pdist2(Xts'*P, X_te_pro, distance);
                    [~, predict_label] = min(dist, [], 2);
                    gzsls = computeAcc(predict_label, Yts, ts_ind)*100;
                    fprintf('[3] %s GZSL seen->all accuracy [X >>> Y <<< A]: %.1f%%\n', dataset, gzsls);
                    H = 2 * gzslu * gzsls / (gzslu + gzsls);
                    disp(['GZSL: H=' num2str(H) ' [X >>> Y <<< A]']);
                    
                    time = toc;
                    if isWrite
                        fprintf(fid,'zsl = %.1f, gzslu = %.1f, gzsls = %.1f, H = %.1f\n', zsl,gzslu,gzsls,H);
                        resultMat(countRst,1) = zsl;
                        resultMat(countRst,2) = gzslu;
                        resultMat(countRst,3) = gzsls;
                        resultMat(countRst,4) = H;
                    end
                    k = k+1;
                end
                if isWrite
                    fclose(fid);
                end
            end
        end
    end
end