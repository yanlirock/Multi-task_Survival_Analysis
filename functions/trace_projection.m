%% FUNCTION trace_projection
%   solves the Trace-norm projection problem.
% 
%% OBJECTIVE
%   argmin_X = 0.5 \|X - L\| + alpha/2 \|L\| 
%
%
%% RELATED PAPERS
%   [1] Cai et. al. A Singlar Value Thresholding Algorihtm for Matrix
%   Completion.


function [L_hat L_tn] = trace_projection(L, alpha)

[d1 d2] = size(L);

if (d1 > d2)
    
    [U S V] = svd(L, 0);
    
    thresholded_value = diag(S) - alpha / 2;
    
    diag_S = thresholded_value .* ( thresholded_value > 0 );
    
    L_hat = U * diag(diag_S) * V';
    L_tn = sum(diag_S);
else 

    new_L = L';
    
    [U S V] = svd(new_L, 0);
    
    thresholded_value = diag(S) - alpha / 2;
    
    diag_S = thresholded_value .* ( thresholded_value > 0 );
    
    L_hat = U * diag(diag_S) * V';

    L_hat = L_hat';
    L_tn = sum(diag_S);
end
