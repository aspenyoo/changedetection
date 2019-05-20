function [LL,p_C_hat] = calculate_joint_LL(x,data_E,data_L,model,logflag,nSamples)


% model indices
% encoding = model(1);        % actual noise. 1: VP, 2: FP
% infering = model(2);     % assumed noise. 1: VP, 2: FP, 3: single value
% decision_rule = model(3);   % decision rule. 1: optimal, 2: max

% % data stuff
% nTrials = size(data.rel,1);
% nItems = size(data.rel,2);
% condition = 'Line';
% condition = data.pres2stimuli;

% ===== GET PARAMETER VALUES ======
idx_Lonly = 3; % indices that only apply to the Line condition
x(logflag) = exp(x(logflag));

% calculate LL for ellipse data
xx = x;
xx(idx_Lonly) = [];
[LL, pch] = calculate_LL_diffdisp(xx,data_E,model,[],nSamples);
p_C_hat.Ellipse = pch;

% add it to LL for line data
[ll, pch] = calculate_LL_diffdisp(x,data_L,model,[],nSamples);
LL = LL + ll;
p_C_hat.Line = pch;


