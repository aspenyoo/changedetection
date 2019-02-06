% Use this vonmises function if you have to draw large numbers of samples
% cdfLin is column vector equal to linspace(-pi,pi,divs), divs affects
%   precision of the interpolation

function A = my_vmrnd_pc(out_size,cdf,cdfLin)

% draw values from cdf
temp_rnd = rand(prod(out_size),1);

% do interpolation
out_temp = qinterp1(1/(cdfLin(2) - cdfLin(1)),cdf,temp_rnd);
A = reshape(out_temp,out_size);
