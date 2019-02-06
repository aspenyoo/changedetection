% load all the model predictions for all subjects for a single model

function [BIG_P_C_HAT] = load_P_C_HAT(model_idx)

load param_data_file
load Subj_data_cell

a = model_idx(1);
b = model_idx(2);
c = model_idx(3);
d = model_idx(4);

BIG_P_C_HAT = cell(size(Subj_data_cell));

for i = 1:length(Subj_data_cell)
    load(['LL/LL_' num2str(i) '_' num2str(a) '_' num2str(b) '_' num2str(c) '_' num2str(d) '.mat']);
    BIG_P_C_HAT{i} = P_C_HAT;
end