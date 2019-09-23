function [x,y] = lineCoord(length, rotation)
% function [x,y] = lineCoord(LENGTH,ROTATION) calculates the
% coordinates of a rotated line with specified LENGTH. 
% 
% ========== INPUT VARIABLES ==========
% LENGTH: length of line
% ROTATION: degrees clockwise from vertical. 
% 
% ========== OUTPUT VARIABLES ==========
% X: change in x axis (from the middle of the line)
% Y: change in the y axis (from the middle of the line)
%
% 
% last updated -- Aspen Yoo 01/31/2019

y = length/2*sind(rotation); % bc in psychtoolbox, positive y is downward
x = -length/2*cosd(rotation);
