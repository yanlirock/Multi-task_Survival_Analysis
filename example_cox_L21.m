function example_cox_L21(floder, name_train, name_test,lam_iter,Smallest_lambda_rate,ratepp)
current_path=cd;
Num_lambda=str2num(lam_iter);
smallest_rate=str2double(Smallest_lambda_rate);
addpath(genpath([current_path '/functions/'])); % load function

% tell the direction where it contains train/test data.
dir=strcat(current_path,'/data/',floder); 
load(strcat(dir,name_train,'.mat')); % load training data.
load(strcat(dir,name_test,'.mat')); % load testing data.
d = size(test_cell{1}, 2)-2;  % dimensionality.
num_task =size(test_cell,1);

opts.init = 0;      % guess start point from data. 
opts.tFlag = 1;     % terminate after relative objective value does not changes much.
opts.tol = 10^-4;   % tolerance. 
opts.maxIter = 100; % maximum iteration number of optimization.

%%build the output matrix
sparsity = zeros(Num_lambda, 1);

cindex=zeros(Num_lambda, num_task);
%AUC_matrix=zeros(Num_lambda,num_task);
%contains=zeros(num_task,1);
%F1=zeros(Num_lambda,num_task);

%% TRAIN
%%Initialize the parameter 
B_old = sparse(zeros(d, num_task));


%%Calculate the smallest possible \lambad_1 which will make B=0

max_lambda = L21_maxlambda( train_cell )*str2num(ratepp);

%%pawise wise search for best \lambad_1
lambda = zeros(1,Num_lambda);
for i=1:Num_lambda
    lambda(i)=max_lambda*(smallest_rate)^(i/Num_lambda);
end
log_lam  = log(lambda);

ALL_B=cell(1,Num_lambda);
tic;    
for i = 1: Num_lambda
    dif=1;
    iter=1;
    fprintf('%d\n',i);

    [B funcVal_B] = cox_L21(train_cell, B_old, lambda(i), opts);
    %Least_L21_Standard(X, newY, lambda(i),rho,B_old, opts);
    B=sparse(B);

    % set the solution as the next initial point to get Warm-start. 
    opts.init = 1;
    opts.W0 = B;
    B_old=B;
    sparsity(i) = nnz(sum(B,2 )==0)/d;
    ALL_B{i}=B; %contains all B's with respect to different lambda
end
toc; %output the training time

%% TESTING
for i = 1: Num_lambda
    for jj =1:num_task
    predict=test_cell{jj}(:,3:end)*ALL_B{i}(:,jj);
    cindex(i,jj)=getcindex_cox(predict,test_cell{jj}(:,1),test_cell{jj}(:,2));
    end
end
% the possible based c-index with different regularizer parameter
Max_cindex=max(cindex);
disp(Max_cindex);
save('result_1.mat','cindex','ALL_B')
end

