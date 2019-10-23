function plot_summaryfit(xrange,partM,partSD,modelM,modelSD,partcolor,modelcolor,mult)
%function summaryplot(XRANGE,PARTM,PARTSD,MODELM,MODELSD,PLOTCOLOR) will
%plot error bars using PARTM and PARTSD against a filled area respresenting
%MODELM and MODELSD
%
% Aspen Yoo
% 2/16/15 -- v1

if nargin < 8; mult = []; end
if nargin < 7; modelcolor = 'k'; end
if nargin < 6; partcolor = 'k'; end

% making sure they are all horizontal vectors
partM = partM(:)';
partSD = partSD(:)';
modelM = modelM(:)';
modelSD = modelSD(:)';

xrangeM = xrange;
% getting rid of nans
nanidx = isnan(modelM);
xrangeM(nanidx) = [];
modelM(nanidx) = [];
modelSD(nanidx) = [];

nanidx = isnan(modelSD);
xrangeM(nanidx) = [];
modelM(nanidx) = [];
modelSD(nanidx) = [];

% plotting
hold on; defaultplot;
if isempty(partSD) || sum(partSD == 0)==length(partSD);
    if ~(isempty(partM))
    hplot = plot(xrange,partM);
    set(hplot               ,...
        'Color'     ,partcolor);
    end
else
    herr = errorbar(xrange, partM, partSD);
    set(get(get(herr,'Annotation'),'LegendInformation'),...
                    'IconDisplayStyle','off'); % Exclude line from legend
                
    set(herr                     ,...
        'Color'         , partcolor        ,...
        'Marker'        ,'none'        ,...
        'LineStyle'     ,'none'     ,...
        'LineWidth'     , 1         );
    
    if ~isempty(mult); try, errbar(herr,mult); end, end
    
    % just used so that you can have a nice legend
    filler = plot(xrange(1)*ones(1,2),partM(1)+[-partSD(1) partSD(1)]);
    set(filler              ,...
        'Color'         ,partcolor          ,...
        'LineWidth'     ,1                );
end
if isempty(modelSD) || sum(modelSD == 0) == length(modelSD);
    if ~(isempty(modelM))
        plot(xrange,modelM,'-','Color',modelcolor);
    end
else
    hfill = fill([xrangeM fliplr(xrangeM)],[modelM-modelSD ...
        fliplr(modelM+modelSD)],modelcolor);
    
    set(hfill                    ,...
        'FaceColor'         ,modelcolor        ,...
        'FaceAlpha'         ,0.4     ,...
        'LineStyle'     ,'none'     ,...
        'LineWidth'     , 1         );
end
hold off;


end


