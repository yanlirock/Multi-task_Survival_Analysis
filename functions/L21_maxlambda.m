function [ lambda_max ] = L21_maxlambda( cox_processed )
%L21_MAXLAMBDA Summary of this function goes here
%   Detailed explanation goes here
task_num  = size(cox_processed,1);
dimension = size(cox_processed{1}.X, 2);
W= zeros(dimension,task_num);
grad_W = zeros(dimension, task_num);

for i = 1:task_num
    [ grad_W(:, i)] = gradandorg_neglogparlike(W(:, i),cox_processed{i});
end

lambda_max=max(sqrt(sum(grad_W.^2,2)));


end

    
    function [dl]=gradandorg_neglogparlike(b,cox_processed)
    % Compute log likelihood L
    X=cox_processed.X;
    freq=cox_processed.freq;
    cens=cox_processed.cens;
    atrisk=cox_processed.atrisk;

    obsfreq = freq .* ~cens;
    Xb = X*b;
    r = exp(Xb);
    risksum = flipud(cumsum(flipud(freq.*r)));
    risksum = risksum(atrisk);


    [n,p] = size(X);
    Xr = X .* repmat(r.*freq,1,p);
    Xrsum = flipud(cumsum(flipud(Xr)));
    Xrsum = Xrsum(atrisk,:);
    A = Xrsum ./ repmat(risksum,1,p);
    dl = obsfreq' * (X-A);
    dl = -dl';
end

