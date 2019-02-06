
%% set up data into useable format

subj_ID_cell = {'METEST','POO'};
pres2stimuli_cell = {'Line','Ellipse'};
subj_data_cell = combine_all_data(subj_ID_cell,pres2stimuli_cell);

%% plot psychometric function of current dataset

nBins = 10;
plot_subjdata = 1;
[p_C_hat_mat_subj,HR_mean_subj,FA_mean_subj] = compute_psych_curves(nBins,plot_subjdata);
