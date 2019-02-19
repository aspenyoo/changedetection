function LL = calculate_joint_LL(x,data_E,data_L,model,logflag,nSamples)


% model indices
% encoding = model(1);        % actual noise. 1: VP, 2: FP
infering = model(2);     % assumed noise. 1: VP, 2: FP, 3: single value
decision_rule = model(3);   % decision rule. 1: optimal, 2: max

% % data stuff
% nTrials = size(data.rel,1);
% nItems = size(data.rel,2);
% condition = 'Line';
% condition = data.pres2stimuli;

% ===== GET PARAMETER VALUES ======
if (decision_rule == 1) && (infering == 3);
    idx_Lonly = [3 6];
else
    idx_Lonly = [3];
end
x(logflag) = exp(x(logflag));

% calculate LL for ellipse data
xx = x;
xx(idx_Lonly) = [];
lf = logflag;
lf(idx_Lonly) = [];
LL = calculate_LL(xx,data_E,model,lf,nSamples);

% add it to LL for ellipse data
LL = LL + calculate_LL(x,data_L,model,logflag,nSamples);

