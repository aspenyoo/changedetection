function resp = fun_LL(x,dMat,model,condition,logflag)
if nargin < 5; logflag = []; end
% nSamples = 1;
% if nargin < 5; nSamples = 50; end

% model indices
encoding = model(1);        % actual noise. 1: VP, 2: FP
infering = model(2);        % assumed noise. 1: VP, 2: FP, 3: stimulus, 4: single value
decision_rule = model(3);   % decision rule. 1: optimal, 2: max
decision_noise = model(4);  % decision noise: 0: none, 1: local, 2: global

% start off with lapse
lapserate = x(end);
nTrials = size(dMat,1);
islapse = rand(1,nTrials) < lapserate;
lapserespVec = rand(1,sum(islapse)) > 0.5;      % for lapse trials, flip a coin
resp = nan(length(islapse),1);
resp(islapse) = lapserespVec;

if sum(~islapse) % if there are any trials that did not lapse
% reduce dMat to only include nonlapse trials
dMat = dMat(~islapse,:);

% define data stuff
nTrials = size(dMat,1);
nItems = 4;
Delta = dMat(:,1:nItems);           % amount change for each of four items
Rels = dMat(:,(nItems+1):end);      % reliabilities for each item (1: low, 2: high)

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

if (infering >= 3) && ~(strcmp(condition,'Line') && (infering == 4) && (decision_rule == 2))% if assumed same precision
    Jbar_assumed = x(counter);
    counter = counter+1;
    
    if strcmp(condition,'Line') && (infering == 3)
        Jbar_line_assumed = x(counter);
        counter = counter+1;
    end
end

if (decision_noise) % if there is some type of decision rule
    sigma_d = x(counter);
end
    
if (decision_rule == 1) % if optimal decision rule
    p_change = x(end-1);
else % max rule
    criterion = x(end-1);
end

% ====== CALCULATE P(\HAT{C}==1|\Theta) FOR nSamples SAMPLES =====

% make CDF for interpolating J to Kappa
tempp = load('cdf_table.mat');
% K_interp = tempp.K_interp;
% cdf = tempp.cdf;
k_range = tempp.k_range;
J_lin = tempp.J_lin;
highest_J = tempp.highest_J;
clear tempp

% calculate actual kappa and noisy representations
get_deltas = 1;
[delta_noise, kappa_x_i, kappa_y_i] = generate_representations(encoding);
get_deltas = 0;

if (infering==4) && (decision_rule==2)      % if model is EIM of VIM
    d_i_Mat = abs(bsxfun(@plus,Delta,delta_noise));
else
    if (encoding ~= infering) % if there is a mismatch in generative and inference process
            [~, kappa_x_i, kappa_y_i] = generate_representations(infering);
    end
    
    % the term inside denominator bessel function for d_i
    % sqrt(\kappa_{x,i}^2 + \kappa_{y,i}^2 + 2\kappa_{x,i}\kappa_{y,i}cos(y_i-x_i))
    Kc = bsxfun(@times,2.*kappa_x_i.*kappa_y_i,cos(bsxfun(@plus,Delta,delta_noise))); % note: it is okay to simply add the noise bc it goes through a cos!!
    Kc = sqrt(bsxfun(@plus,kappa_x_i.^2+kappa_y_i.^2,Kc)); % dims: mat_dims
    
    % d_i
    d_i_Mat = bsxfun(@minus,log(besseli(0,kappa_x_i,1).*besseli(0,kappa_y_i,1))+...
        (kappa_x_i+kappa_y_i),log(besseli(0,Kc,1))+Kc); % actually log d_i_Mat
%     Kc(Kc>Lookup(end)) = Lookup(end); % clip large values
%     d_i_Mat = bsxfun(@rdivide,myBessel(kappa_x_i,LookupSpacing,LookupY).*myBessel(kappa_y_i,LookupSpacing,LookupY),...
%         myBessel(Kc,LookupSpacing,LookupY));
end

if (decision_noise == 1);   % if local decision noise
    d_i_Mat = d_i_Mat + randn(size(d_i_Mat)).*sigma_d;
end

if (decision_rule == 1); % if optimal
    p_C_hat = log(sum(exp(d_i_Mat),2))-log(nItems)+log(p_change)-log(1-p_change);  % these values are actually log(d), not p_C_hat
    if (decision_noise == 2); p_C_hat = p_C_hat + randn(size(p_C_hat)).*sigma_d; end    % if global dec noise
    p_C_hat = p_C_hat > 0; %1;      % respond 1 if log(d) > log(1)
else
    p_C_hat = max(d_i_Mat,[],2);                % these values are actually log(d), not p_C_hat
    if (decision_noise == 2);  % if global dec noise
        p_C_hat = p_C_hat + randn(size(p_C_hat)).*sigma_d; 
    end   
    p_C_hat = p_C_hat > criterion;  % respond 1 if max(log(d_i)) > criterion
end

resp(~islapse) = p_C_hat;
end

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

        mat_dims = [nTrials nItems];
        
        if any(precision == [1 2]) % VP, FP
            % fill in matrix J_mat according to trial precisions
            Jbar_mat = Rels;
            Jbar_mat(Rels==1) = Jbar_low;
            Jbar_mat(Rels==2) = Jbar_high;
        end
        
        switch precision % the precision at which kappas are generated
            case 1      % VP
                J_x_mat = gamrnd(Jbar_mat./tau,tau);
                if strcmp(condition,'Line') % if second stimulus set were lines
                    J_y_mat = gamrnd(Jbar_line./tau,tau,mat_dims);
                else
                    J_y_mat = gamrnd(Jbar_mat./tau,tau);
                end
            case 2      % FP
                J_x_mat = Jbar_mat;
                if strcmp(condition,'Line') % if second stimulus set were lines
                    J_y_mat = Jbar_line*ones(mat_dims);
                else
                    J_y_mat = Jbar_mat;
                end
            case {3,4}
                J_x_mat = Jbar_assumed*ones(mat_dims);
                if strcmp(condition,'Line') && (precision == 3)
                    J_y_mat = Jbar_line_assumed*ones(mat_dims);
                else
                    J_y_mat = J_x_mat;
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
        
        if size(kappa_x,2) ~= nItems
            kappa_x = kappa_x';
            kappa_y = kappa_y';
        end
        
        if (get_deltas) % only used in generative stage
            
            try
            noise_x = circ_vmrnd(0,kappa_x);
            noise_y = circ_vmrnd(0,kappa_y);
            catch
                dMat
                mat_dims
                J_x_mat
                xi
                kappa_x
                kappa_y
            end
            

%             
%             % get closest kappa idx
%             idx_kappa_x = interp1(K_interp,1:length(K_interp),kappa_x_temp,'nearest');
%             idx_kappa_y = interp1(K_interp,1:length(K_interp),kappa_y_temp,'nearest');
%             
%             noise_x = randi(size(cdf,2),prod(mat_dims),1); % get random row indices (to sample from cdf)
%             noise_x = (noise_x-1)*size(cdf,1) + idx_kappa_x(:);
%             noise_x = cdf(noise_x);
%             noise_x = reshape(noise_x,mat_dims);
%             
%             noise_y = randi(size(cdf,2),prod(mat_dims),1); % get random row indices
%             noise_y = (noise_y-1)*size(cdf,1) + idx_kappa_y(:);
%             noise_y = cdf(noise_y);
%             noise_y = reshape(noise_y,mat_dims);
            
            % get difference between noise
            delta_noise = noise_x-noise_y;
        else
            delta_noise = [];
        end
        
    end
end
