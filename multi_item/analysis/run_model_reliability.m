% Run a specific model on the data
%
%                               USAGE
% subj_data_idx: subject ID number
% param_data_file: name of file containing all parameter ranges
%
% encoding: Type of encoding
%           1-> VP encoding
%           2-> FP encoding
% variability: Type of assumed variability
%           1-> VP variability
%           2-> FP variability
%           3-> Kappa proportional to average reliability per trial
%           4-> Single assumed kappa
% decision rule: Type of decision rule
%           1-> Optimal decision rule
%           2-> Max rule
% decision noise: Type of noise on decision variable
%           1-> No noise
%           2-> Local decision noise
%           3-> Global decision noise
%           4-> Both global and local decision noise (NOT IN PAPER)

function [LL,P_C_HAT] = run_model_reliability(subj_data_idx,model_idx)

% model indices
encoding = model_idx(1); 
variability = model_idx(2);
decision_rule = model_idx(3);
decision_noise = model_idx(4);

fprintf('Encoding: %g Variability: %g Decision Rule: %g Decision Noise: %g Subject: %g\n',encoding,variability,decision_rule,decision_noise,subj_data_idx)

% load parameter ranges & data
A = load('../data/param_data_file.mat');
load('../data/Subj_data_cell.mat');
Subj_data_cell = revert_data(Subj_data_cell);

J_vec = A.J_vec;
J_vec_FP = A.J_vec_FP;
theta_vec = A.theta_vec;
crit_vec = A.crit_vec;
prior_vec = A.prior_vec;
local_d_noise_vec = A.local_d_noise_vec;
global_d_noise_vec = A.global_d_noise_vec;
num_samples = A.num_samples;

% Reset randomn number seed
rng('shuffle')

% If testing FP, don't need theta and use J_vec_FP
if encoding == 2
    theta_vec = 1;
    J_vec = J_vec_FP;
end

% J_assumed_vec is the same as J_vec
J_ass_vec = J_vec;

% If NOT testing single-resource, don't need J_assumed
if variability ~= 4
    J_ass_vec = 1;
end

% If NOT using max rule, don't need fitted criterion parameter
if decision_rule ~= 2
    crit_vec = 1;
    crit_idx = 1;
else
    prior_vec = 1;
end

% take log ratio of prior vector for easier addition later 
% shift dimensions for use in bsxfun
logprior_vec = shiftdim(log(prior_vec./(1-prior_vec)),-4);

% set size & numTrials from data, assuming all same set size
N = Subj_data_cell{subj_data_idx}(1,5);
num_trials = size(Subj_data_cell{subj_data_idx},1);

% Only keep needed decision noise
if decision_noise == 1
    local_d_noise_vec = 1; global_d_noise_vec = 1;
elseif decision_noise == 2
    global_d_noise_vec = 1;
    local_d_noise_mat = bsxfun(@plus,local_d_noise_vec,zeros(num_samples,1,1,N,num_trials));
elseif decision_noise == 3
    local_d_noise_vec = 1;
    global_d_noise_mat = bsxfun(@plus,shiftdim(global_d_noise_vec,-1),zeros(num_samples,1,1,1,num_trials));
else
    local_d_noise_mat = bsxfun(@plus,local_d_noise_vec,zeros(num_samples,1,1,N,num_trials));
    global_d_noise_mat = bsxfun(@plus,shiftdim(global_d_noise_vec,-1),zeros(num_samples,1,1,1,num_trials));
end

% J_low, J_high locations initialized to speed up encoding in VR
J_init = zeros(num_samples,N,num_trials);
delta_noise = J_init;

% Lookup tables for modified bessel function and J to Kappa equation
Lookup(:,1) = linspace(0,700.92179,3500)';
Lookup(:,2) = besseli(0,Lookup(:,1));
LookupY = Lookup(:,2);
LookupSpacing = 1/(Lookup(2,1)-Lookup(1,1));
cdfLin = linspace(-pi,pi,1000)';
highest_J = 700.92179;
K_interp = [0 logspace(log10(1e-3),log10(700.92179),1999)];

% make CDF for interpolating J to Kappa
cdf = make_cdf_table(K_interp,cdfLin);

k_range_temp = linspace(0,700.92179,6001)';
J_range = k_range_temp.*(besseli(1,k_range_temp)./besseli(0,k_range_temp));
J_lin = linspace(min(J_range),max(J_range),6001)';
k_range = interp1(J_range,k_range_temp,J_lin);
highest_J = max(J_range);

% Counters and things
t = []; 

% Get data into [rel], [delta] format
Subj_data = Subj_data_cell{subj_data_idx};
Data_rel = Subj_data(:,39:(38+N)); % reliabilities
Data_delta = -Subj_data(:,56:(55+N))*(pi/90) + Subj_data(:,64:(63+N))*(pi/90); % y-x
C_hat = Subj_data(:,2);

% SORT DATA
% First, sort each trial. Low to high reliability
[rel_sorted I_t] = sort(Data_rel,2);
Delta_sorted = zeros(size(Data_delta));
for j = 1:size(rel_sorted,1)
    Delta_sorted(j,:) = Data_delta(j,I_t(j,:));
end

% Now, sort by number of reliabilities. Low to high
rels = unique(Data_rel);
high_num = sum(Data_rel==rels(2),2);
[high_sorted I] = sort(high_num); I_LATER = I;
low_sorted = N-high_sorted;
delta = Delta_sorted(I,:);
rel_sorted = rel_sorted(I,:);
low_rel = rel_sorted == rels(1);
high_rel = rel_sorted == rels(2);
C_hat = C_hat(I);
delta = bsxfun(@plus,shiftdim(delta',-1),J_init);
low_rel = bsxfun(@plus,shiftdim(low_rel',-1),J_init);
high_rel = bsxfun(@plus,shiftdim(high_rel',-1),J_init);
low_rel = low_rel==1;
high_rel = high_rel==1;

% create LL matrix. J_LOW x J_HIGH x THETA x J_ASS x CRITERIA x LOCAL D
% NOISE x GLOBAL D NOISE x PRIOR
LL = zeros(length(J_vec),length(J_vec),length(theta_vec),length(J_ass_vec),...
    length(crit_vec),length(local_d_noise_vec),length(global_d_noise_vec),length(prior_vec));
P_C_HAT = zeros(size(Subj_data,1));

time_idx = 1;
num_samples
% loop over parameters
for J_high_idx = 1:length(J_vec)

    for J_low_idx = 1:J_high_idx % J_low < J_high by definition
        tic;
        
        for theta_idx = 1:length(theta_vec)
            
            % set current parameter values
            J_high = J_vec(J_high_idx);
            J_low = J_vec(J_low_idx);
            theta_val = theta_vec(theta_idx);
            
            % generate kappas and internal rep. for all trials, stimuli,
            % and samples. They are 3D matrices-> SAMPLES x N x TRIALS
            get_deltas = 1;
            
            switch encoding
                case 1 % encode with VR
                    
                    [delta_noise kappa_x kappa_y] = encode_VR();
                    
                case 2 % encode with ER
                    if (variability == 2) || (theta_idx == 1)
                        [delta_noise kappa_x kappa_y] = encode_ER();
                    end
            end
            get_deltas = 0;
            % loop over J_assumed
            for J_ass_idx = 1:length(J_ass_vec)
                J_ass = J_ass_vec(J_ass_idx);
                
                % generate assumed variability and compute decision var.
                switch variability
                    case 1 % Assume VP variability
                        
                        % optimized for VP/FP
                        
                        % if encoded with VP
                        if encoding == 1
                            kappa_x_i = kappa_x;
                            kappa_y_i = kappa_y;
                            Kc = sqrt(kappa_x_i.^2+kappa_y_i.^2+2*kappa_x_i.*kappa_y_i.*cos(delta+delta_noise));
                            Kc(Kc>Lookup(end,1)) = Lookup(end,1); % clip large values
                            d = myBessel(kappa_x_i,LookupSpacing,LookupY).*myBessel(kappa_y_i,LookupSpacing,LookupY)./myBessel(Kc,LookupSpacing,LookupY);
                            
                            % if encoded with FP
                        else
                            [tmp kappa_x_i kappa_y_i] = encode_VR();
                            Kc = sqrt(kappa_x_i.^2+kappa_y_i.^2+2*kappa_x_i.*kappa_y_i.*cos(delta+delta_noise));
                            Kc(Kc>Lookup(end,1)) = Lookup(end,1);
                            d = myBessel(kappa_x_i,LookupSpacing,LookupY).*myBessel(kappa_y_i,LookupSpacing,LookupY)./myBessel(Kc,LookupSpacing,LookupY);
                            
                        end
                        
                    case 2 % Assume FP variability
                        
                        % optimize for VP/FP
                        % if encoded with FP
                        if encoding == 1
                            [tmp kappa_x_i kappa_y_i] = encode_ER(); %#ok<NASGU>
                            Kc = sqrt(bsxfun(@times,2*kappa_x_i.^2,1+cos(delta+delta_noise)));
                            Kc(Kc>Lookup(end,1)) = Lookup(end,1);
                            d = bsxfun(@rdivide,myBessel(kappa_x_i,LookupSpacing,LookupY).^2,myBessel(Kc,LookupSpacing,LookupY));
                            
                            % if encoded with FP
                        else
                            Kc = sqrt(bsxfun(@times,2*kappa_x.^2,cos(delta+delta_noise)+1));
                            Kc(Kc>Lookup(end,1)) = Lookup(end,1);
                            d = bsxfun(@rdivide,myBessel(kappa_x,LookupSpacing,LookupY).^2,myBessel(Kc,LookupSpacing,LookupY));
                        end
                        
                    case 3 % average variability
                        
                        % optimize if FP
                        if encoding == 1
                            kappa_x_i = bsxfun(@plus,mean(kappa_x,2),zeros(size(kappa_x)));
                            kappa_y_i = bsxfun(@plus,mean(kappa_y,2),zeros(size(kappa_y)));
                            Kc = sqrt(kappa_x_i.^2+kappa_y_i.^2+2*kappa_x_i.*kappa_y_i.*cos(delta+delta_noise));
                            Kc(Kc>Lookup(end,1)) = Lookup(end,1);
                            d = bsxfun(@times,myBessel(kappa_x_i(:,1,:),LookupSpacing,LookupY).*myBessel(kappa_y_i(:,1,:),LookupSpacing,LookupY),1./myBessel(Kc,LookupSpacing,LookupY));
                        else % if VP
                            kappa_x_i = bsxfun(@plus,mean(kappa_x,2),zeros(size(kappa_x)));
                            Kc = sqrt(bsxfun(@times,2*kappa_x_i.^2,1+cos(delta+delta_noise)));
                            Kc(Kc>Lookup(end,1)) = Lookup(end,1);
                            d = bsxfun(@rdivide,myBessel(kappa_x_i,LookupSpacing,LookupY).^2,myBessel(Kc,LookupSpacing,LookupY));
                        end
                        
                    case 4 % single variability
                        kappa_ass = J_to_Kappa(J_ass);
                        kappa_ass_bessel = besseli(0,kappa_ass);
                        Kc = sqrt(2)*kappa_ass*sqrt(1+cos(delta+delta_noise));
                        Kc(Kc>Lookup(end,1)) = Lookup(end,1);
                        
                        denom = reshape(LookupY(round(LookupSpacing*Kc(:) + 1)),size(Kc));
                        
                        d = (kappa_ass_bessel)./denom;
                end
                
                % generate decisions
                switch decision_rule
                    case 1 % full decision rule
                        
                        switch decision_noise
                            case 1
                                d_temp = permute(shiftdim(d,-2),[3 1 2 4 5]);
                                d_temp_3 = bsxfun(@plus,log(sum(d_temp,4))-log(N),logprior_vec);
                                p_C_hat_temp = shiftdim((mean(d_temp_3>0,1)),1);
                                clear d_temp_3
                                p_C_hat = permute(p_C_hat_temp,[4 1 2 5 3]);
                                clear p_C_hat_temp
                            case 2 % local decision noise
                                d_temp = permute(shiftdim(d,-2),[3 1 2 4 5]);
                                d_temp_2 = exp(bsxfun(@plus,log(d_temp),normrnd(0,local_d_noise_mat)));
                                clear d_temp
                                d_temp_3 = bsxfun(@plus,log(sum(d_temp_2,4))-log(N),logprior_vec);
                                clear d_temp_2
                                p_C_hat_temp = shiftdim((mean(d_temp_3>0,1)),1);
                                clear d_temp_3
                                p_C_hat = permute(p_C_hat_temp,[4 1 2 5 3]);
                            case 3 % global decision noise
                                d_temp = permute(shiftdim(d,-2),[3 1 2 4 5]);
                                d_temp_2 = bsxfun(@plus,log(sum(d_temp,4))-log(N),normrnd(0,global_d_noise_mat));
                                clear d_temp
                                d_temp_3 = bsxfun(@plus,d_temp_2,logprior_vec);
                                clear d_temp_2
                                p_C_hat_temp = shiftdim(mean(d_temp_3>0,1),1);
                                clear d_temp_3
                                p_C_hat = permute(p_C_hat_temp,[4 1 2 5 3]);
                            case 4 % local and global decision noise
                                d_temp = permute(shiftdim(d,-2),[3 1 2 4 5]);
                                d_temp_1 = exp(bsxfun(@plus,log(d_temp),normrnd(0,local_d_noise_mat)));
                                clear d_temp
                                d_temp_2 = bsxfun(@plus,log(sum(d_temp_1,4))-log(N),normrnd(0,global_d_noise_mat));
                                clear d_temp d_temp_1 d
                                p_C_hat_temp = shiftdim(mean(bsxfun(@plus,d_temp_2,logprior_vec)>0,1),1);
                                p_C_hat = permute(p_C_hat_temp,[4 1 2 5 3]);
                                clear p_C_hat_temp
                        end
                        
                    case 2 % max rule
                        crit_vec_shift = shiftdim(crit_vec,-2);
                        d_temp = bsxfun(@minus,max((log(d)-log(N)),[],2),crit_vec_shift)>0;
                        p_C_hat = squeeze(mean(d_temp));
                end
                
                p_resp = bsxfun(@times,(C_hat == 1),p_C_hat) + bsxfun(@times,(C_hat == 0),(1-p_C_hat));
                
                % Fix very small p_resp
                p_resp(p_resp==0) = 1/(100*num_samples);
                
                % record log(p_resp) for parameter combination, summing
                % over trials
                curr_LL = shiftdim(sum(log(p_resp),1),1);
                
                % save maximizing P_C_HAT
                if max(curr_LL(:))>max(LL(LL~=0))
                    [X I] = max(curr_LL(:));
                    Subj_data_temp = Subj_data(I_LATER,:);
                    Subj_data_temp(:,2) = squeeze(p_C_hat(:,I));
                    P_C_HAT = Subj_data_temp;
                end
                
                LL(J_low_idx,J_high_idx,theta_idx,J_ass_idx,:,:,:,:) = curr_LL;
                
            end
            
        end
        
        % display time remaining. NOT VERY ACCURATE
        t = toc;
        fprintf('Time left: about %g min\n',(sum((length(J_vec)):-1:0)-time_idx)*t/60);
        time_idx = time_idx+1;
        
    end
    
end

save(['./LL/LL_' num2str(subj_data_idx) '_' num2str(encoding) '_' num2str(variability) '_' num2str(decision_rule) '_' num2str(decision_noise)],'LL','P_C_HAT','-v7.3');

% helper functions

    function [delta_noise kappa_x kappa_y] = encode_VR()
        kappa_x = J_init;
        kappa_y = J_init;
        delta_noise = kappa_x;
        
        % generate J_low and J_high samples
        J_low_sample = gamrnd(J_low/theta_val,theta_val,[num_samples 2*N]);
        J_high_sample = gamrnd(J_high/theta_val,theta_val,[num_samples 2*N]);
        
        % resample kappas too high
        high_count_low = J_low_sample>highest_J;
        high_count_high = J_high_sample>highest_J;
        while (sum(high_count_low(:)) + sum(high_count_high(:))) > 0
            J_low_sample(high_count_low) = gamrnd(J_low/theta_val,theta_val,[sum(high_count_low(:)) 1]);
            J_high_sample(high_count_high) = gamrnd(J_high/theta_val,theta_val,[sum(high_count_high(:)) 1]);
            
            high_count_low = J_low_sample>highest_J;
            high_count_high = J_high_sample>highest_J;
        end
        
        % convert J to kappa
        kappa_low_sample = qinterp1(1/(J_lin(2)-J_lin(1)),k_range,J_low_sample);
        kappa_high_sample = qinterp1(1/(J_lin(2)-J_lin(1)),k_range,J_high_sample);
        
        % calculate noise
        noise_low_i = circ_vmrnd(0,kappa_low_sample,1);
        noise_high_i = circ_vmrnd(0,kappa_high_sample,1);
        
        for kk = 1:num_trials
            low_num_curr = low_sorted(kk);
            high_num_curr = N-low_num_curr;
            
            BigY = [kappa_low_sample(:,(N+1):(N+low_num_curr)) kappa_high_sample(:,(N+1):(N+high_num_curr))...
                noise_low_i(:,(N+1):(N+low_num_curr)) noise_high_i(:,(N+1):(N+high_num_curr))];
            shift = mod(kk,size(BigY,1))+1;
            BigY_shifted = [BigY(shift:end,:); BigY(1:(shift-1),:)];
            
            kappa_low_x = kappa_low_sample(:,1:low_num_curr);
            kappa_low_y = BigY_shifted(:,1:(low_num_curr));
            idx = low_num_curr;
            
            kappa_high_x = kappa_high_sample(:,1:high_num_curr);
            kappa_high_y = BigY_shifted(:,(idx+1):(idx+high_num_curr));
            idx = idx + high_num_curr;
            
            noise_low = noise_low_i(:,1:low_num_curr)-BigY_shifted(:,(idx+1):(idx+low_num_curr));
            idx = idx + low_num_curr;
            noise_high = noise_high_i(:,1:high_num_curr)-BigY_shifted(:,(idx+1):(idx+high_num_curr));
            
            kappa_x(:,:,kk) = [kappa_low_x kappa_high_x];
            kappa_y(:,:,kk) = [kappa_low_y kappa_high_y];
            
            if get_deltas
                delta_noise(:,:,kk) = [noise_low noise_high];
            else
                delta_noise = [];
            end
            
        end
    end
    function [delta_noise kappa_x kappa_y] = encode_ER()
        kappa_x = J_init;
        delta_noise = kappa_x;
        
        % Convert from J to Kappa
        kappa_low = J_to_Kappa(J_low);
        kappa_high = J_to_Kappa(J_high);
        
        % Get "nearest" kappa for interpolation for my_vmrnd_pc
        kappa_low_idx = interp1(K_interp,1:length(K_interp),kappa_low,'nearest');
        kappa_high_idx = interp1(K_interp,1:length(K_interp),kappa_high,'nearest');
        
        kappa_low = K_interp(kappa_low_idx);
        kappa_high = K_interp(kappa_high_idx);
        
        % assign high and low kappas
        kappa_x(low_rel) = kappa_low;
        kappa_x(high_rel) = kappa_high;
        kappa_y = kappa_x;
        
        if get_deltas
            % generate delta noise
            noise_temp_low = my_vmrnd_pc([sum(low_rel(:)) 2],cdf(kappa_low_idx,:),linspace(0,1,size(cdf,2)));
            noise_temp_high = my_vmrnd_pc([sum(high_rel(:)) 2],cdf(kappa_high_idx,:),linspace(0,1,size(cdf,2)));
            
            % get noise difference for high and low kappa
            noise_temp_low_diff = noise_temp_low(:,1) - noise_temp_low(:,2);
            noise_temp_high_diff = noise_temp_high(:,1) - noise_temp_high(:,2);
            
            % save noise in its proper location
            delta_noise(low_rel) = noise_temp_low_diff;
            delta_noise(high_rel) = noise_temp_high_diff;            
        else
            delta_noise = [];
        end
    end
end
