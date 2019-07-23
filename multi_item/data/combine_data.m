function trialMat = combine_data(subjid, pres2stimuli)

if (ismac)
    dir_str = '/Volumes/GoogleDrive/';
else
    dir_str = 'G:/';
end
dir_str = [dir_str 'My Drive/Research/VSTM/Aspen Luigi - Reliability in VWM/'...
    'Exp 5 - Keshvari replication and extension/experiment output/'];
temp_dir = dir([dir_str subjid '/']);

trialMat = [];
for j = 1:length(temp_dir)
    if ~isempty(regexp(temp_dir(j).name,sprintf('Reliability_%s',pres2stimuli),'once'))
        load([dir_str subjid '/' temp_dir(j).name])
        
        % deleting trials not completed yet
        exited_idx = find(TrialMat(:,4)==0,1,'first');
        TrialMat(exited_idx:end,:) = [];
        
        trialMat = [trialMat; TrialMat];
    end
end

save(sprintf('data/combined_data/%s_%s_combined.mat',subjid,pres2stimuli),'trialMat','subjid','pres2stimuli')