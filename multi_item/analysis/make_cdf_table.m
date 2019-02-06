% precompute a the CDF of the bessel function for faster simulation

function [cdf] = make_cdf_table(kappa_lin,theta_lin)

kappas = kappa_lin;
thetas = theta_lin;
mu = 0;
cdf = zeros(length(kappas),length(thetas));

for i = 1:length(kappas)
    cdf(i,:) = (2*pi/length(thetas))*cumtrapz(1/(2*pi*besseli(0,kappas(i)))*exp(kappas(i)*cos(thetas-mu)));
    
    %fix cumsum problem
    % if a number is repeated 
    % hacks to fix cdf
    
    cdf(i,:) = abs(normrnd(cdf(i,:),1e-8));
    cdf(i,1) = 0;
    a = max(cdf(i,:));
    cdf(i,:) = cdf(i,:)/a;
    cdf(i,:) = sort(cdf(i,:));
    
    while length(unique(cdf(i,:))) < length(cdf(i,:))
        cdf(i,:) = abs(normrnd(cdf(i,:),1e-8));
        cdf(i,1) = 0;
        a = max(cdf(i,:));
        cdf(i,:) = cdf(i,:)/a;
        cdf(i,:) = sort(cdf(i,:));
    end
    
    % re-interpolate to have uniform x
    cdf(i,:) = interp1(cdf(i,:),theta_lin,linspace(0,1,length(thetas)));
    
end
