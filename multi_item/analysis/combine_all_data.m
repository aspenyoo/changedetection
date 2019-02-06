% combine data from subjects in subj_vec
% 
% original Keshvari 2012
% edited by Aspen Yoo 2019

function [Subj_data_cell] = combine_all_data(subj_ID_cell,pres2stimuli_cell)
%COMBINE_ALL_DATA combines all data from subjects with their ids in cell
%SUGJ_ID_CELL. 
% 
% PRES2STIMULI: cell of strings corresponding tot the pres2timuli for each
% SUBJ_ID_CELL

nSubj = length(subj_ID_cell);
Subj_data_cell = cell(1,nSubj);
dir_str = ['G:/My Drive/Research/VSTM/Aspen Luigi - Reliability in VWM/'...
    'Exp 5 Keshvari replication and extension/experiment output/'];
for isubj = 1:nSubj
    
    curr_subj = subj_ID_cell{isubj};
    pres2stimuli = pres2stimuli_cell{isubj};
    temp_dir = dir([dir_str curr_subj '/']);
    
    TrialMatTemp = [];
    
    for j = 3:length(temp_dir)
        
        if ~isempty(regexp(temp_dir(j).name,sprintf('Reliability_%s',pres2stimuli),'once'))
            
            load([dir_str curr_subj '/' temp_dir(j).name])
            TrialMatTemp = [TrialMatTemp; TrialMat];
            
        end
        
    end
    
    Subj_data_cell{isubj} = TrialMatTemp;
    
end

Subj_data_cell = convert_data(Subj_data_cell);
for isubj = 1:nSubj
    pres2stimuli = pres2stimuli_cell{isubj};
    
    Subj_data_cell{isubj}.subj_ID = subj_ID_cell{isubj}; % subject id
    Subj_data_cell{isubj}.condition = pres2stimuli; % condition of pres2stimuli
    if strcmp(pres2stimuli,'Line')
        Subj_data_cell{isubj}.ecc_second = ones(size(Subj_data_cell{isubj}.ecc_second));
    end
end