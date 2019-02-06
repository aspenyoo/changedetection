% linear interpolation to convert J to Kappa

function [kappa] = J_to_Kappa(J_val)

% precompute inverse Fisher information function (inverse J)
k = linspace(0,700.92179,6001);
J = k.*(besseli(1,k)./besseli(0,k));

kappa = interp1(J',k',J_val);