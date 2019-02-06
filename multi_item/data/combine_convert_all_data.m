% combine data from subjects in subj_vec
% 
% original Keshvari 2012
% edited by Aspen Yoo 2019

function [Subj_data_cell] = combine_convert_all_data(subj_ID_cell,pres2stimuli_cell)
%COMBINE_ALL_DATA combines all data from subjects with their ids in cell
%SUGJ_ID_CELL. 
% 
% PRES2STIMULI: cell of strings corresponding tot the pres2timuli for each
% SUBJ_ID_CELL

nSubj = length(subj_ID_cell);
Subj_data_cell = cell(1,nSubj);
for isubj = 1:nSubj
    
    curr_subj = subj_ID_cell{isubj};
    pres2stimuli = pres2stimuli_cell{isubj};
    
    Subj_data_cell{isubj} = combine_data(curr_subj,pres2stimuli);
    
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