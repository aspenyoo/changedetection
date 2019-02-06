% combine data from subjects in subj_vec

function [Subj_data_cell] = combine_all_data(subj_ID_cell)

Subj_data_cell = cell(1,size(subj_ID_cell,1));

for i = 1:size(subj_ID_cell,1)
    
    curr_subj = subj_ID_cell{i,:};
    temp_dir = dir(['./output/' curr_subj '/']);
    
    TrialMatTemp = [];
    
    for j = 3:length(temp_dir)
        
        if ~isempty(regexp(temp_dir(j).name,'Experimental', 'once'))
            
            load(['./output/' curr_subj '/' temp_dir(j).name])
            TrialMatTemp = [TrialMatTemp; TrialMat];
            
        end
        
    end
    
    Subj_data_cell{i} = TrialMatTemp;
    
end

Subj_data_cell = convert_data(Subj_data_cell);