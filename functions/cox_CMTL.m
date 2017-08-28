%% FUNCTION cox_CMTL
%   Convex-relaxed Clustered Multi-Task Learning with Cox prprotional Loss.
%% Code starts here
function [W, funcVal,funcVal_cox, M] = cox_CMTL(cox_processed,rho1, rho2, k, W_old, opts)

if nargin <4
    error('\n Inputs: cox_processed,rho1, rho2,and k should be specified!\n');
end

if nargin <5
    opts = [];
end

if rho2<=0 || rho1<=0
    error('rho1 and rho2 should both greater than zero.');
end

% if exist('mosekopt','file')==0
%     error('Mosek is not found. Please install Mosek first. \n')
% end

% initialize options.
%opts=init_opts(opts);

task_num  = size(cox_processed,1);
funcVal = [];
funcVal_cox= [];

eta = rho2 / rho1;
c = rho1 * eta * (1 + eta);

M0 = speye (task_num) * k / task_num;

bFlag=0; % this flag tests whether the gradient step only changes a little

Wz= W_old;
Wz_old = W_old;
Mz = M0;
Mz_old = M0;

t = 1;
t_old = 0;


iter = 0;
gamma = 1;
gamma_inc = 2;

while iter < opts.maxIter
    alpha = (t_old - 1) /t;
    
    Ws = (1 + alpha) * Wz - alpha * Wz_old;
    Ms = (1 + alpha) * Mz - alpha * Mz_old;
    % compute function value and gradients of the search point
    %gWs  = gradVal_eval(Ws, rho1);
    %Fs   = funVal_eval  (Ws, rho1);
    [gWs, gMs, Fs] = gradVal_eval (Ws, Ms);
    
    while true
        %         [Wzp l1c_wzp] = l1_projection(Ws - gWs/gamma, 2 * rho2 / gamma);
        %         Fzp = funVal_eval  (Wzp, rho1);
        Wzp = Ws - gWs/gamma;
        [Mzp ,Mzp_Pz, Mzp_DiagSigz ] = singular_projection (Ms - gMs/gamma, k);
        [Fzp, F_cox] = funVal_eval (Wzp, Mzp_Pz, Mzp_DiagSigz);
        
        %Fzp_gamma = Fs + trace(delta_Wzp' * gWs) + gamma/2 * norm(delta_Wzp, 'fro')^2;
        
        delta_Wzs = Wzp - Ws;
        delta_Mzs = Mzp - Ms;
        
        r_sum = (norm(delta_Wzs, 'fro')^2 + norm(delta_Mzs, 'fro')^2)/2;
        
        
        %         Fzp_gamma = Fs + trace( (delta_Wzs)' * gWs) ...
        %             + trace( (delta_Mzs)' * gMs) ...
        %             + gamma/2 * norm(delta_Wzs, 'fro')^2 ...
        %             + gamma/2 * norm(delta_Mzs, 'fro')^2;
        Fzp_gamma = Fs + sum(sum( delta_Wzs .* gWs)) ...
            + sum(sum( delta_Mzs .* gMs)) ...
            + gamma/2 * norm(delta_Wzs, 'fro')^2 ...
            + gamma/2 * norm(delta_Mzs, 'fro')^2;
        
        if (r_sum <=eps)
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
    Mz_old = Mz;
    Mz = Mzp;
    
    funcVal = cat(1, funcVal, Fzp);
    funcVal_cox = cat(1, funcVal_cox, F_cox);
    
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
M = Mzp;

% private functions

    function [Mzp, Mzp_Pz, Mzp_DiagSigz ] = singular_projection (Msp, k)
        [EVector, EValue] = eig(Msp);
        Pz = real(EVector);  diag_EValue = real(diag(EValue));
        %DiagSigz = SingVal_Projection(diag_EValue, k); % use mosek
        DiagSigz = bsa_ihb(diag_EValue, ones(size(diag_EValue)), k, ones(size(diag_EValue)));
        Mzp = Pz * diag(DiagSigz) *Pz';
        Mzp_Pz = Pz;
        Mzp_DiagSigz = DiagSigz;
    end
    
    % the negtive partial_likelihood of Cox model
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

    % the gradient of negtive partial_likelihood of Cox model
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

    function [grad_W, grad_M, funcVal] = gradVal_eval(W ,M)
        IM = (eta * speye(task_num) + M);
        invEtaMWt = IM\W';
        
        grad_W = [];
        for t_ii = 1:task_num
            dl=gradandorg_neglogparlike(W(:,t_ii),cox_processed{t_ii});
            grad_W = cat(2, grad_W, dl);
        end

        grad_W = grad_W + 2 * c * invEtaMWt';   %W component
        grad_M = - c * (W' * W / IM /IM );      %M component
        
        funcVal = 0;
        for iii = 1: task_num
            funcVal = funcVal+neglogparlike(W(:,iii),cox_processed{iii});
            %funcVal = funcVal + 0.5 * norm (Y{i} - X{i}' * W(:, i))^2;
        end 
        funcVal = funcVal + c * trace( W * invEtaMWt);
    end

    function [funcVal,funcValorg] = funVal_eval (W, M_Pz, M_DiagSigz)
        invIM = M_Pz * (diag( 1./(eta + M_DiagSigz))) * M_Pz';
        invEtaMWt = invIM * W';
        
        funcVal = 0;
        for i = 1: task_num
            funcVal = funcVal+neglogparlike(W(:,i),cox_processed{i});
            %funcVal = funcVal + 0.5 * norm (Y{i} - X{i}' * W(:, i))^2;
        end  
        funcValorg = funcVal ;
        funcVal = funcVal  + c * trace( W * invEtaMWt);
    end

end