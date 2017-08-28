%% FUNCTION Least_Trace
%   Trace-Norm Regularized Learning with Cox Loss.

%% Code starts here
function [W, funcVal] = Cox_Trace(cox_processed, W0, rho1, opts)


if nargin <3
    error('\n Inputs: cox_processed, W0, rho1, should be specified!\n');
end

if nargin <4
    opts = [];
end

% initialize options.
%opts=init_opts(opts);

if isfield(opts, 'rho_L2')
    rho_L2 = opts.rho_L2;
else
    rho_L2 = 0;
end

task_num  = size(cox_processed,1);
dimension = size(cox_processed{1}.X, 2);
funcVal = [];


bFlag=0; % this flag tests whether the gradient step only changes a little


Wz= W0;
Wz_old = W0;

t = 1;
t_old = 0;
iter = 0;
gamma = 1;
gamma_inc = 2;

while iter < opts.maxIter
    alpha = (t_old - 1) /t;
    
    Ws = (1 + alpha) * Wz - alpha * Wz_old;
    
    % compute function value and gradients of the search point
    gWs  = gradVal_eval(Ws);
    Fs   = funVal_eval  (Ws);
    
    while true
        [Wzp Wzp_tn] = trace_projection(Ws - gWs/gamma, 2 * rho1 / gamma);
        Fzp = funVal_eval  (Wzp);
        
        delta_Wzp = Wzp - Ws;
        r_sum = norm(delta_Wzp, 'fro')^2;
        %Fzp_gamma = Fs + trace(delta_Wzp' * gWs) + gamma/2 * norm(delta_Wzp, 'fro')^2;
        Fzp_gamma = Fs + sum(sum(delta_Wzp .* gWs)) + gamma/2 * norm(delta_Wzp, 'fro')^2;
        
        if (r_sum <=1e-20)
            bFlag=1; % this shows that, the gradient step makes little improvement
            break;
        end
        
        if (Fzp <= Fzp_gamma)
            break;
        else
            gamma = gamma * gamma_inc;
        end
    end
    
    Wz_old = Wz;
    Wz = Wzp;
    
    %funcVal = cat(1, funcVal, Fzp + rho1 * sum( svd(Wzp, 0) ));
    funcVal = cat(1, funcVal, Fzp + rho1 * Wzp_tn);
    
    if (bFlag)
        % fprintf('\n The program terminates as the gradient step changes the solution very small.');
        break;
    end
    
    % test stop condition.
    switch(opts.tFlag)
        case 0
            if iter>=2
                if (abs( funcVal(end) - funcVal(end-1) ) <= opts.tol)
                    break;
                end
            end
        case 1
            if iter>=2
                if (abs( funcVal(end) - funcVal(end-1) ) <=...
                        opts.tol* funcVal(end-1))
                    break;
                end
            end
        case 2
            if ( funcVal(end)<= opts.tol)
                break;
            end
        case 3
            if iter>=opts.maxIter
                break;
            end
    end
    
    iter = iter + 1;
    t_old = t;
    t = 0.5 * (1 + (1+ 4 * t^2)^0.5);
    
end

W = Wzp;


% private functions
    function [grad_W] = gradVal_eval(W)
        grad_W = zeros(dimension, task_num);
        
        for i = 1:task_num
            [ grad_W(:, i)] = gradandorg_neglogparlike(W(:, i),cox_processed{i});
        end
        grad_W = grad_W + rho_L2 * 2 * W;
    end

    function [funcVal] = funVal_eval (W)
        funcVal = 0;
        for i = 1: task_num
            funcVal = funcVal + neglogparlike(W(:, i),cox_processed{i});
        end
        funcVal = funcVal + rho_L2 * norm(W, 'fro')^2;
    end

end

function [L]=neglogparlike(b,cox_processed)
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
    L = obsfreq'*(Xb - log(risksum));
    L = -L;
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