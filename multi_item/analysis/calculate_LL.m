function [LL,p_C_hat] = calculate_LL(x,data,model,logflag,nSamples)
if nargin < 4; logflag = []; end
if nargin < 5; nSamples = 200; end

% model indices
encoding = model(1);        % actual noise. 1: VP, 2: FP
infering = model(2);     % assumed noise. 1: VP, 2: FP, 3: single value
decision_rule = model(3);   % decision rule. 1: optimal, 2: max

% data stuff
nTrials = size(data.rel,1);
nItems = size(data.rel,2);
condition = data.pres2stimuli;

% ===== GET PARAMETER VALUES ======
counter = 3;
x(logflag) = exp(x(logflag));
Jbar_high = x(1);
Jbar_low = x(2);

if strcmp(condition,'Line');
    Jbar_line = x(counter);
    counter = counter+1;
end

if (encoding == 1); % if VP
    tau = x(counter);
    counter = counter+1;
end

if (decision_rule == 1) % if optimal decision rule
    if (infering == 3) % if assumed same precision
        Jbar_assumed = x(counter);
        counter = counter+1;
        
        if strcmp(condition,'Line')
            Jbar_line_assumed = x(counter);
        end
    end
    p_change = x(end);
else % max rule
    criterion = x(end);
end


% ====== CALCULATE P(\HAT{C}==1|\Theta) FOR nSamples SAMPLES =====

% make CDF for interpolating J to Kappa
tempp = load('cdf_table.mat');
K_interp = tempp.K_interp;
cdf = tempp.cdf;
k_range = tempp.k_range;
J_lin = tempp.J_lin;
highest_J = tempp.highest_J;
clear tempp

% highest_J = 700.92179;
% cdfLin = linspace(-pi,pi,1000)';
% K_interp = [0 logspace(log10(1e-3),log10(highest_J),1999)];
% cdf = make_cdf_table(K_interp,cdfLin);
% tic;
% k_range = linspace(0,700.92179,6001)';
% J_range = k_range.*(besseli(1,k_range)./besseli(0,k_range));
% J_lin = linspace(min(J_range),max(J_range),6001)';
% k_range = interp1(J_range,k_range,J_lin);
% highest_J = max(J_range);
% toc

% calculate actual kappa and noisy representations
get_deltas = 1;
[delta_noise, kappa_x_i, kappa_y_i] = generate_representations(encoding);
get_deltas = 0;

if (infering==3) && (decision_rule==2)      % if model is ESM
    d_i_Mat = abs(delta_noise);
else
    if (encoding ~= infering) % if there is a mismatch in generative and inference process
        if (infering == 3);
            [kappa_x_i, kappa_y_i] = deal(Jbar_assumed);
            if (strcmp(condition,'Line'))
                kappa_y_i = Jbar_line_assumed;
            end
        else
            [~, kappa_x_i, kappa_y_i] = generate_representations(infering);
        end
    end
    
    % the term inside denominator bessel function for d_i
    % sqrt(\kappa_{x,i}^2 + \kappa_{y,i}^2 + 2\kappa_{x,i}\kappa_{y,i}cos(y_i-x_i))
    Kc = bsxfun(@times,2.*kappa_x_i.*kappa_y_i,cos(bsxfun(@plus,data.Delta,delta_noise))); % note: it is okay to simply add the noise bc it goes through a cos!!
    Kc = sqrt(bsxfun(@plus,kappa_x_i.^2+kappa_y_i.^2,Kc)); % dims: mat_dims
    
    % d_i
    d_i_Mat = bsxfun(@minus,log(besseli(0,kappa_x_i,1).*besseli(0,kappa_y_i,1))+...
        (kappa_x_i+kappa_y_i),log(besseli(0,Kc,1))+Kc); % actually log d_i_Mat
%     Kc(Kc>Lookup(end)) = Lookup(end); % clip large values
%     d_i_Mat = bsxfun(@rdivide,myBessel(kappa_x_i,LookupSpacing,LookupY).*myBessel(kappa_y_i,LookupSpacing,LookupY),...
%         myBessel(Kc,LookupSpacing,LookupY));
end

if (decision_rule == 1); % if optimal
    p_C_hat = log(sum(exp(d_i_Mat),2))-log(nItems)+log(p_change)-log(1-p_change);  % the value is actually of d, not p_C_hat
    p_C_hat = p_C_hat > 1;      % respond 1 if d > 1
else
    p_C_hat = max(d_i_Mat,[],2);
    p_C_hat = p_C_hat > criterion;  % respond 1 if max d_i > criterion
end
p_C_hat = mean(p_C_hat,3); % get average across samples
p_C_hat(p_C_hat==0) = eps; % set zero p_C to something very low
p_C_hat(p_C_hat==1) = 1-eps;

% calculate LL across trials
LL = data.resp'*log(p_C_hat) + (1-data.resp)'*log(1-p_C_hat);

    % ================================================================
    %                      HELPER FUNCTIONS
    % ================================================================
    function [delta_noise, kappa_x, kappa_y] = generate_representations(precision)
        % PRECISION
        % 1 - VP: kappa_x and kappa_y will be of dimension [nTrials,nItems,nSamples]
        % 2 - FP: kappa_x and kappa_y will be of dimension [nTrials,nItems]
        % 3 - SP: kappa_x and kappa_y will be of dimension [nTrials,nItems]
        % 
        % CONDITION
        % 'Ellipse': kappa_x and kappa_y have the same structure
        % 'Line': all items in kappa_y are drawn from Jbar_line (in VP or FP).
        %
        % note: Jbar_line == 0 means ellipse condition. change if actual
        % Jbar_line = 0 is allowed
        %
        % ======= OUTPUT VARIABLE =====
        % DELTA_NOISE: matrix of dimension [nTrials,nItems,nSamples]

        mat_dims = [nTrials,nItems,nSamples];
        %     [delta_noise, kappa_x, kappa_y] = deal(nan(nTrials,nItems,nSamples));
        n_high_vec = 0:nItems;
        rels = unique(data.rel);            % unique reliabilities across experiment
        
        idx_high = [0 nan(1,nItems+1)];
        for n_high = n_high_vec;
            idx_high(n_high+2) = find(sum(data.rel == rels(2),2)==n_high,1,'last');
        end
        
        if (precision == 3) % SP
            J_x_mat = Jbar_assumed*ones(nTrials,nItems);
            if strcmp(condition,'Line')
                J_y_mat = Jbar_line_assumed*ones(nTrials,nItems);
            else
                J_y_mat = J_x_mat;
            end
        else % VP, FP
            % fill in matrix J_mat according to trial precisions
            Jbar_mat = nan(nTrials,nItems);
            for ihigh = 1:length(n_high_vec);
                n_low = nItems-n_high_vec(ihigh);         % number of high rel items
                idx_start = idx_high(ihigh)+1;       % which row starts this n_high
                idx_stop = idx_high(ihigh+1);        % end of this thing
                
                Jbar_mat(idx_start:idx_stop,1:n_low) = Jbar_low;
                Jbar_mat(idx_start:idx_stop,(n_low+1):nItems) = Jbar_high;
            end
        end
        
        switch precision % the precision at which kappas are generated
            case 1      % VP
                Jbar_mat = repmat(Jbar_mat,[1 1 nSamples]);
                J_x_mat = gamrnd(Jbar_mat./tau,tau);
                if strcmp(condition,'Line') % if second stimulus set were lines
                    J_y_mat = gamrnd(Jbar_line./tau,tau,mat_dims);
                else
                    J_y_mat = gamrnd(Jbar_mat./tau,tau);
                end
            case 2      % FP
                J_x_mat = Jbar_mat;
                if strcmp(condition,'Line') % if second stimulus set were lines
                    J_y_mat = Jbar_line;
                else
                    J_y_mat = Jbar_mat;
                end
        end
        
        % set kappas too high to highest J (alternatively can resample, as
        % keshvari did)
        J_x_mat(J_x_mat > highest_J) = highest_J;
        J_y_mat(J_y_mat > highest_J) = highest_J;
        
        % convert J to kappa
        xi = 1/diff(J_lin(1:2))*J_x_mat+1;
        kappa_x = k_range(round(xi));
        xi = 1/diff(J_lin(1:2))*J_y_mat+1;
        kappa_y = k_range(round(xi));
%         kappa_x = qinterp1(1/(J_lin(2)-J_lin(1)),k_range,J_x_mat);
%         kappa_y = qinterp1(1/(J_lin(2)-J_lin(1)),k_range,J_y_mat);
        
        if (get_deltas) % only used in generative stage
            % get matrices in correct dimensions
            if (length(size(kappa_x)) ~= length(mat_dims))
                kappa_x_temp = bsxfun(@times,kappa_x,ones(mat_dims));
            else
                kappa_x_temp = kappa_x;
            end
            if (length(size(kappa_y)) ~= length(mat_dims))
                kappa_y_temp = bsxfun(@times,kappa_y,ones(mat_dims));
            else
                kappa_y_temp = kappa_y;
            end
            
            % SOME OTHER WAY (1.5 SECOND FOR 200 SAMPLES)
            % get closest kappa idx
            idx_kappa_x = interp1(K_interp,1:length(K_interp),kappa_x_temp,'nearest');
            idx_kappa_y = interp1(K_interp,1:length(K_interp),kappa_y_temp,'nearest');
            
            noise_x = randi(size(cdf,2),prod(mat_dims),1); % get random row indices
            noise_x = size(cdf,2)*(idx_kappa_x(:)-1)+noise_x; % make them linear indices
            noise_x = cdf(noise_x);
            noise_x = reshape(noise_x,mat_dims);
            
            noise_y = randi(size(cdf,2),prod(mat_dims),1); % get random row indices
            noise_y = size(cdf,2)*(idx_kappa_y(:)-1)+noise_y; % make them linear indices
            noise_y = cdf(noise_y);
            noise_y = reshape(noise_y,mat_dims);
            
%             % CALCULATE NOISE KESHVARI WAY (12 SECONDS)
%             % get closest kappa idx
%             idx_kappa_x = interp1(K_interp,1:length(K_interp),kappa_x_temp,'nearest');
%             idx_kappa_y = interp1(K_interp,1:length(K_interp),kappa_y_temp,'nearest');
%         
%             % calculate noise
%             noise_x = my_vmrnd_pc(size(kappa_x_temp),cdf(idx_kappa_x,:),linspace(0,1,size(cdf,2)));
%             noise_y = my_vmrnd_pc(size(kappa_y_temp),cdf(idx_kappa_y,:),linspace(0,1,size(cdf,2)));
            
%             % CALCULATE NOISE DUMB WAY(takes about 4 seconds for 200 samples)
%             noise_x = circ_vmrnd(0,kappa_x_temp);
%             noise_y = circ_vmrnd(0,kappa_y_temp);
            
            % get difference between noise
            delta_noise = noise_x-noise_y;
        else
            delta_noise = [];
        end
        %     rels = unique(data.rel);
        %     idx_low = find(data.rel == rels(1));
        %     idx_high = find(data.rel == rels(2));
        %     [i_low, j_low] = ind2sub(size(data.rel),idx_low);
        %     [i_high,j_high] = ind2sub(size(data.rel),idx_high);
        
    end
end
