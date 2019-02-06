
%% set up data into useable format

subj_ID_cell = {'METEST','POO'};
pres2stimuli_cell = {'Line','Ellipse'};
Subj_data_cell = combine_all_data(subj_ID_cell,pres2stimuli_cell);

save('Subj_data_cell.mat','Subj_data_cell')

%% plot psychometric function of current dataset

% close all
nBins = 6;
plot_subjdata = 1;
[delta_bin_vec,p_C_hat_mat_subj,HR_subj,FA_subj] = compute_psych_curves(nBins,plot_subjdata);

% plot indvl psychometric curves
nSubj = size(FA_subj,1);
N_vec = (0:4)';


% plot
figure
for isubj = 1:nSubj;
    
    % proportion report change
    subplot(nSubj,2,2*isubj-1)
    hold on;
    for irel = 1:length(N_vec);
        plot(delta_bin_vec,squeeze(p_C_hat_mat_subj(isubj,irel,:)));
    end
    ylim([0 1]);
    if isubj == nSubj; 
        xlabel('Magnitude of change in radians')
        legend([repmat('N_H=',length(N_vec),1) num2str(N_vec)])
    end;
    if isubj == 1; title('Probability report "Change"'), end;
    defaultplot
    
    % HR and FA
    subplot(nSubj,2,2*isubj)
    hold on; 
    plot(N_vec,HR_subj(isubj,:)); % hit rate
    plot(N_vec,FA_subj(isubj,:)); % false alarm
    ylim([0 1]);
    if (isubj == 1); title('Hit and false alarm rates'), end
    if (isubj == nSubj);
        legend('Hit Rate','False Alarm Rate')
        xlabel('Number of high reliability items (N_H)')
    end
    defaultplot
end
