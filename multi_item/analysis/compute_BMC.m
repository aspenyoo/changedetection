% compute model posterior for bayesian model comparison
% computed per model over all subjects

function bars = compute_BMC(model_idx,subj_id_cell,pres2stimuli_cell)

% load parameter ranges & data
load('param_data_file.mat');
% load('Subj_data_cell.mat');
nSubj = length(subj_id_cell);
bars = NaN(nSubj,1);

% model indices
encoding = model_idx(1);
variability = model_idx(2);
decision_rule = model_idx(3);
decision_noise = model_idx(4);

% get the penalty associated with each free parameter
% for the J parameters, only half of the parameter range is used
J_range_FP = log(max(J_vec_FP)-min(J_vec_FP))+log((max(J_vec_FP)-min(J_vec_FP))/2);
J_range_VP = log(max(J_vec)-min(J_vec))+log((max(J_vec)-min(J_vec))/2);
J_range_FP_full = log(max(J_vec_FP)-min(J_vec_FP)); % for models with single assumed kappa
J_range_VP_full = log(max(J_vec)-min(J_vec)); % ditto
prior_range = log(max(prior_vec)-min(prior_vec));
local_d_range = log(max(local_d_noise_vec)-min(local_d_noise_vec));
global_d_range = log(max(global_d_noise_vec)-min(global_d_noise_vec));
theta_range = log(max(theta_vec)-min(theta_vec));
crit_range = log(max(crit_vec)-min(crit_vec));

% for each subject
for isubj = 1:nSubj
    
    subjid = subj_id_cell{isubj};
    pres2stimuli = pres2stimuli_cell{isubj};
    
    % load the current subject and model
    load(['LL/LL_' subjid '_' pres2stimuli '_' num2str(encoding) '_' ...
        num2str(variability) '_' num2str(decision_rule) '_' ...
        num2str(decision_noise) '.mat']);

    % exponentiate the log likelihood after subtracting max value
    LL_max = max(LL(LL~=0));
    LL_exp = exp(LL-LL_max);
    clear temp_LL;
    clear temp_pun;
    
    % encoding type (variable precision or fixed precision)
    switch encoding
        case 1 % VP
            temp_LL = sqz_trap(theta_vec,sqz_trap_J(J_vec,LL_exp));
            temp_pun = -J_range_VP-theta_range;
            
        case 2 % FP
            temp_LL = sqz_trap_J(J_vec_FP,LL_exp);
            temp_pun = -J_range_FP;
    end
    
    % decision noise, only local (2) and global (3) supported separately
    switch decision_noise
        case 2
            temp_LL = sqz_trap(local_d_noise_vec,temp_LL);
            temp_pun = temp_pun - local_d_range;
        case 3
            temp_LL = sqz_trap(global_d_noise_vec,temp_LL);
            temp_pun = temp_pun - global_d_range;
        case 4
            error('\nsimultaneous global and local noise unsupported!')
    end
    
    % assumed kappa, only "single assumed kappa" adds a free parameter
    if variability == 4
        % which parameter vector we used depends on the encoding
        switch encoding
            case 1 % VP
                temp_LL = sqz_trap(J_vec,temp_LL);
                temp_pun = temp_pun-J_range_VP_full;
                
            case 2 % FP
                temp_LL = sqz_trap(J_vec_FP,temp_LL);
                temp_pun = temp_pun-J_range_FP_full;
        end
    end
    
    % decision rule
    switch decision_rule
        case 1 % optimal decision, adds the decision prior parameter
            temp_LL = sqz_trap(prior_vec,temp_LL);
            temp_pun = temp_pun-prior_range;
        case 2 % max rule, adds criterion parameter
            temp_LL = sqz_trap(crit_vec,temp_LL);
            temp_pun = temp_pun-crit_range;
    end
    
    % get the log posterior by taking log, adding back in the max value,
    % and adding in punishment terms
    bars(isubj) = log(temp_LL) + LL_max + temp_pun;
    
end
end

% helper function to do trapezoidal sum and squeeze output
function out = sqz_trap(x,y)

out = squeeze(trapz(x,y));

if size(out,1)==1
    out = out';
end

end

% same as sqz_trap but deals with non-square shape of parameter space
% associated with the J parameter (J_low < J_high)
function out = sqz_trap_J(x,y)
q = size(y);
out = zeros(q(2:end));

for i = 1:length(x)
    out(i,:,:,:,:,:,:,:) = shiftdim(trapz(x(1:i),y(1:i,i,:,:,:,:,:,:),1),1);
end

out = squeeze(trapz(x,out));

end