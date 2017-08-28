%% file example_cox_CMTL.m
% this file shows the usage of cox_CMTL.m function

function example_cox_CMTL(floder, name_train,name_test,lam_iter,clus_num,Smallest_lambda_rate)
current_path=cd;
Num_lambda=str2num(lam_iter);
smallest_rate=str2double(Smallest_lambda_rate);
addpath(genpath([current_path '/functions/'])); % load function
clus_num=str2num(clus_num);

% tell the direction where it contains train/test data.
dir=strcat(current_path,'/data/',floder); 
load(strcat(dir,name_train,'.mat')); % load training data.
load(strcat(dir,name_test,'.mat')); % load testing data.

task_num = size(train_cell,1); % total task number.

opts.init = 0;      % guess start point from data.
opts.tFlag = 1;     % terminate after relative objective value does not changes much.
opts.tol = 10^-5;   % tolerance.
opts.maxIter = 1000; % maximum iteration number of optimization.
epsilon=10^-5;

rho_1 = 10;
%rho_2 = 10^0;
W_old = zeros((size(test_cell{1},2)-2),task_num);

%kmCMTL_OrderedModel = zeros(size(W));
%OrderedTrueModel = zeros(size(W));

Cindex=zeros(Num_lambda,task_num);
TYPE = zeros(task_num,Num_lambda);
%% testing
for ii=1:Num_lambda
    rho_2=rho_1^(9-ii);
    [W_learn, funcVal,funcVal_cox, M_learned]= cox_CMTL(train_cell, rho_1, rho_2, clus_num,W_old,opts);
    for jj =1:task_num
        predict=test_cell{jj}(:,3:end)*W_learn(:,jj);
        Cindex(ii,jj)=getcindex_cox(predict,test_cell{jj}(:,1),test_cell{jj}(:,2));
    end
    W_old=W_learn;
    [U,S,V] = svd(M_learned);
    tS=S(1:clus_num,:);
    [M,I] = max(abs(tS*V));
    TYPE(:,ii)=I';
end

% shows the best possible Cindex with different parameter
maxc=max(Cindex);
disp(maxc);

end

