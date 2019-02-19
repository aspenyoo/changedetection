% Get the parameter values that maximize log likelihood for a particular
% subject and model

function [a_i b_i c_i d_i e_i f_i] = get_max_params(subj_idx,model_idx)

load param_data_file
load Subj_data_cell

a = model_idx(1);
b = model_idx(2);
c = model_idx(3);
d = model_idx(4);

i = subj_idx;

load(['LL/LL_' num2str(i) '_' num2str(a) '_' num2str(b) '_' num2str(c) '_' num2str(d) '.mat']);
LL(LL==0) = min(LL(:));
[x I] = max(LL(:));
size(LL)
[a_i, b_i, c_i, r2, r3, d_i, e_i, f_i] = ind2sub(size(LL),I);

