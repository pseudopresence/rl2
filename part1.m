%% Setup
clc;
clear all;
close all;
dbstop if error;
dbstop if infnan;

%% Part I - MDP

%% World Setup

stream0 = RandStream('mt19937ar','Seed',0);
%RandStream.setGlobalStream(stream0);
RandStream.setDefaultStream(stream0);

Rot90 = [
     0 +1;
    -1  0
];

% Action representation: Idx mapped to dX, dY by this table

Actions = [
    +1  0; % 1 E >
     0 -1; % 2 S V
    -1  0; % 3 W <
     0 +1; % 4 N ^
];
NActions = size(Actions, 1);

MapWidth = 4;
MapHeight = 3;
MapSize = [MapWidth MapHeight];
NStates = prod(MapSize);
GoalState = 7;

stateFromPos = @(P) (P(1) - 1) + MapWidth * (P(2) - 1) + 1;
posFromState = @(S) [mod((S - 1),MapWidth) + 1, floor((S - 1)/MapWidth) + 1];


%% Walls

% Compute state transition matrix

function T = computeStateTransitionTable()
    T = zeros([NStates, NActions]);

    for LA = 1:NActions
        for LS = 1:NStates
            PS2 = posFromState(LS);
            NP = Actions(LA,:) + PS2;
            NX = NP(1);
            NY = NP(2);

            if (NX < 1 || NX > MapWidth || NY < 1 || NY > MapHeight)
                T(LS, LA) = LS;
            else
                T(LS, LA) = stateFromPos(NP);
            end
        end
    end
end

% [State x Action] -> State
StateTransitionTable = computeStateTransitionTable();

% Make final state absorbing
StateTransitionTable(GoalState, :) = repmat(GoalState, [NActions 1]);

%% Visualisation

% Policy representation: [State] -> Action
% from the goal state
StartPolicy = floor(rand(NStates, 1) * 4) + 1;

% Arrows for rendering state transitions
Arrow(1, :, :) = [
    0.25 0;
    0.75 0;
    0.625 -0.125;
    0.75 0;
    0.625 +0.125;
    0.75 0;
];
% Rotate for other directions
for A = 2:4
    for L = 1:size(Arrow, 2)
        Arrow(A,L,:) = Rot90^(A-1) * squeeze(Arrow(1,L,:));
    end
end

% Arrows for rendering policies
% 0 G X
ActionGlyphs(1, 1, :, :) = [-1 +1; -1 +1]';
ActionGlyphs(2, 1, :, :) = [-1 +1; +1 -1]';
% 1 E >
ActionGlyphs(1, 2, :, :) = [-1 +1; -1 +1]';
ActionGlyphs(2, 2, :, :) = [+1  0; -1  0]';
% 2 S V
ActionGlyphs(1, 3, :, :) = [-1  0; +1  0]';
ActionGlyphs(2, 3, :, :) = [+1 -1; +1 -1]';
% 3 W < 
ActionGlyphs(1, 4, :, :) = [+1 -1; +1 -1]';
ActionGlyphs(2, 4, :, :) = [+1  0; -1  0]';
% 4 N ^
ActionGlyphs(1, 5, :, :) = [-1  0; +1  0]';
ActionGlyphs(2, 5, :, :) = [-1 +1; -1 +1]';
% Rescale
ActionGlyphs = 0.25 * ActionGlyphs;

VizMinX = 0.5;
VizMaxX = MapWidth+0.5;
VizMinY = 0.5;
VizMaxY = MapHeight+0.5;

function drawWalls()
    line([VizMinX VizMaxX], [VizMinY VizMinY], 'Color', 'red');
    line([VizMinX VizMaxX], [VizMaxY VizMaxY], 'Color', 'red');
    line([VizMinX VizMinX], [VizMinY VizMaxY], 'Color', 'red');
    line([VizMaxX VizMaxX], [VizMinY VizMaxY], 'Color', 'red');
end

function drawGlyph(Glyph, P)
    XS = squeeze([Glyph(1:2:end,1), Glyph(2:2:end,1)])';
    YS = squeeze([Glyph(1:2:end,2), Glyph(2:2:end,2)])';
    line(P(1) + XS, P(2) + YS, 'Color', 'green');
end

function drawStateTransitions()
    [NStates NActions] = size(StateTransitionTable);
    for LA = 1:NActions
        for LS = 1:NStates
            PS2 = posFromState(LS);
            if (StateTransitionTable(LS, LA) ~= LS)
                drawGlyph(squeeze(Arrow(LA,:,:,:)), PS2);
            end
        end
    end
end

function drawActionImpl(Glyphs, X, Y, A)
    XS = squeeze(Glyphs(1, A+1,:,:));
    YS = squeeze(Glyphs(2, A+1,:,:));
    line(X + XS, Y + YS, 'Color', 'blue');
end

function drawPolicy(Policy)
    [NStates] = numel(Policy);
    
    for LS = 1:NStates
        PS2 = posFromState(LS);
        X = PS2(1);
        Y = PS2(2);
        drawActionImpl(ActionGlyphs, X, Y, Policy(LS));
    end
end

function vizPolicy(Fig, V, Policy)
     figure(Fig);
        imagesc(reshape(V, MapSize)');
        colormap('gray');
        drawWalls();
        drawPolicy(Policy);
    axis([VizMinX, VizMaxX, VizMinY, VizMaxY], 'xy', 'equal');
end

figure(1);
    drawWalls();
    drawStateTransitions();
    drawPolicy(StartPolicy);
axis([VizMinX, VizMaxX, VizMinY, VizMaxY], 'xy', 'equal');
writeFigureEPS('StartingPolicy.eps');

reward = @(S, A, S2) -1 * (S ~= GoalState);

%% We implement state transition probabilities PP(a, s, s') as a function
%% PP(s, a) -> [States, Probs]; which returns a list of possible next
%% states and their probabilities.

NormalStateTransitions = @(S, A) deal(StateTransitionTable(S, A), 1);

%% Value iteration

function [Policy] = computeGreedyPolicy(V, StateTransitions, Reward, Discount)
    Policy = zeros([NStates, 1]);
    for LS = 1:NStates
        Q = zeros([NActions, 1]);
        for LA = 1:NActions;
            [S2 Pr] = StateTransitions(LS, LA);
            for LI = 1:size(S2, 2)
                Q(LA) = Q(LA) + Pr(LI) * (Reward(LS, LA, S2(LI)) + Discount * V(S2(LI)));
            end
        end
        [~, LA] = max(Q);
        Policy(LS) = LA;
    end
end

function [NV NQ] = valueIterationStep(V, StateTransitions, Reward, Discount)
    NV = zeros([NStates, 1]);
    NQ = zeros([NStates, NActions]);
    for LS = 1:NStates
        Q = zeros([NActions, 1]);
        for LA = 1:NActions;
            [S2 Pr] = StateTransitions(LS, LA);
            for LI = 1:size(S2, 2)
                Q(LA) = Q(LA) + Pr(LI) * (Reward(LS, LA, S2(LI)) + Discount * V(S2(LI)));
            end
        end
        [V2 ~] = max(Q);
        NQ(LS, :) = Q';
        NV(LS) = V2;
    end
end

function [V Q] = valueIteration(Fig, Discount, StateTransitions, MaxPolicyIterations)
    V = zeros([NStates 1]);
    for Iter = 1:MaxPolicyIterations
        % Evaluate policy
        [NewV Q] = valueIterationStep(V, StateTransitions, reward, Discount);

        % Compute greedy policy
        Policy = computeGreedyPolicy(NewV, StateTransitions, reward, Discount);

        if (all(V == NewV))
            break;
        end
        V = NewV;

        vizPolicy(Fig, V, Policy);
    end
    fprintf('Value Iteration: Iterations before policy convergence: %d\n', Iter);
end
fprintf('Value Iteration\n');
Discount = 1;
MaxPolicyIterations = 1000;
[~, Q] = valueIteration(4, Discount, NormalStateTransitions, MaxPolicyIterations);
writeFigureEPS('ValueIteration.eps');