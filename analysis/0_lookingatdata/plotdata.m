function plotdata(data)
% plotdata(subjids) gives rudementary psychometric plots of
% data to see how subjects are doing in detection task
%
%
% aspen.yoo
% April 18, 2016

nCond = length(data);
% get rid of all no-change trials
for icond = 1:nCond;
    blah = data{icond};
    blah(blah(:,1) == 0,:) = [];
    data{icond} = blah;
end


[stimlevels, trialnums, nresps] = conditionSeparator(data);


nLevels = length(stimlevels{1});
conditions = 1:nCond;

colorMat = aspencolors(nCond,'blue');

hold on
for icond = 1:nCond;
    cond = conditions(icond);
    
    for ilevel = 1:nLevels;
        currplot = plot(stimlevels{cond}(ilevel), nresps{cond}(ilevel)/trialnums{cond}(ilevel),'.');
        set(currplot,'MarkerSize', trialnums{cond}(ilevel),...
            'Color',       colorMat(cond,:));
        if ilevel ~= nLevels; set(get(get(currplot,'Annotation'),'LegendInformation'),...
                'IconDisplayStyle','off'); end % Exclude line from legend
    end
    defaultplot;
    axis([-40 40 0 1])
    set(gca,'Ytick',[0 0.5 1],'Xtick',[-40 0 40])
    ylabel('p(respond same)');
    xlabel('orientation change (deg)');
    % 	    legend(legendMat{conditions})
end
