function removetxtspaces(model,subjid,filepath)
if nargin < 3; filepath = 'analysis/4_realdata/fits/'; end

filename = [filepath 'paramfit_model' num2str(model) '_subj' subjid '.txt'];

% Read the file as cell string line by line
fid = fopen(filename,'r');
% FileName = fopen(fid);
if fid < 0, error('Cannot open file: %s', filename); end
Data = textscan(fid, '%s', 'delimiter', '\n', 'whitespace', '');
fclose(fid);

% remove empty lines
idxsMat = [];
for i = 1:10; % for each digit (1:10 -1 = 0:9)
    tempidx = cellfun(@(x) min([find(x == num2str(i-1)) Inf]),Data{1},'UniformOutput',false); % find the earliest index of that digit
    idxsMat = [idxsMat tempidx];    % contatinate that into a matrix of first instance of the digit
end
idxMat = min(cell2mat(idxsMat),[],2);       % find the first digit in that line
data = Data{1};

% get rid of any empty lines (i.e., lines where there are no digits)
data(idxMat == Inf,:) = [];
idxMat(idxMat == Inf) = [];

% get rid of m
idxMat = mat2cell(idxMat,ones(1,length(idxMat)));
data = cellfun(@(x,y) x(y:end),data,idxMat,'UniformOutput',false);

% Write the cell string
fid = fopen(filename, 'w');
fprintf(fid, '%s\r\n', data{:});
fclose(fid);

