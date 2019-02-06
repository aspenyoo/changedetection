function graphProgress(windowPtr,breaknum,PC)

ScreenNumber=max(Screen('Screens'));       % use external Screen if exists
[w, h]=Screen('WindowSize', ScreenNumber);  % Screen resolution

% settings for the graph shown between blocks
graphwidth = round(w/3); % half of graph width
graphheight = round(h/5); % half of graph height
origin = [w/2-graphwidth/2 h/2+graphheight];
dx = graphwidth./breaknum; % equally spaced out to fill size of graph at end of exp.

% rescaling PC to size of graph
pc = PC*graphheight;
    
% axes
Screen('DrawLine',windowPtr,255*ones(1,3),origin(1), origin(2),origin(1)+graphwidth,origin(2),2); % x-axis
Screen('DrawLine',windowPtr,255*ones(1,3),origin(1), origin(2),origin(1),origin(2)-graphheight,2); % y-axis
Screen('DrawLine',windowPtr,175*ones(1,3),origin(1), origin(2)-0.5*graphheight,origin(1)+graphwidth,origin(2)-0.5*graphheight,1); % x-axis

% axis labels
Screen('TextSize',windowPtr,24);
Screen('DrawText',windowPtr,'percent correct per block',w/2-200,origin(2)+15,[255 255 255]); % xlabel
Screen('DrawText',windowPtr,'chance',origin(1)+graphwidth+10,origin(2)-0.5*graphheight-24,175*ones(1,3)); % chance line label

% draw lines connecting performance for each block
og = origin;
dc = nan(2,length(pc));
dc(:,1) = [og(1),og(2)-pc(1)];
for ipc = 2:length(pc) 
    dc(:,ipc) = [og(1)+dx og(2)-pc(ipc)];
    Screen('DrawLine',windowPtr,255*ones(1,3),dc(1,ipc-1),dc(2,ipc-1),dc(1,ipc),dc(2,ipc)); % y-axis
    og(1) = og(1) + dx;
end

% draw dots indicating performance for each block
Screen('DrawDots',windowPtr,dc,7,255*ones(1,3),[],1)