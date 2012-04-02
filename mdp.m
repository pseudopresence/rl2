function mdp()

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
% TODO make X/Y component the last, the remove the squeeze()
% TODO remove the transpose
% TODO generate with rotation matrix
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
    % TODO ugh
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
                % TODO put action index at end...
                drawGlyph(squeeze(Arrow(LA,:,:,:)), PS2);
            end
        end
    end
end

function drawActionImpl(Glyphs, X, Y, A)
    % TODO change X,Y to P
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

% writeFigureEPS('StartingPolicy.pdf');

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
            % TODO - make sure it makes sense and is computing the exact
            % same thing, but return Q from valueIteration instead of V
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

        % refresh;
        % pause(0.1);
    end
    fprintf('Value Iteration: Iterations before policy convergence: %d\n', Iter);
end
fprintf('Value Iteration\n');
Discount = 1;
MaxPolicyIterations = 1000;
[~, Q] = valueIteration(4, Discount, NormalStateTransitions, MaxPolicyIterations);
% writeFigureEPS('NormalValueIteration.pdf');

function [Z2 Pr] = Observations(S, A)
    [S2 Pr] = NormalStateTransitions(S, A);
    PS1 = sum(Pr(S2 == S));
    
    Z2 = [    0   1];
    Pr = [1-PS1 PS1];
end

function vizBel(Fig, Bel)
     figure(Fig);
        imagesc(reshape(Bel, MapSize)');
        colormap('gray');
        drawWalls();
    axis([VizMinX, VizMaxX, VizMinY, VizMaxY], 'ij', 'equal');
end

function [X] = sampleDiscrete(Xs, Ps)
    C = cumsum(Ps);
    R = rand(1);
    I = find(C > R, 1);
    % If non-zero prob of R==0.0, C >= R might select sample with P=0.
    % If non-zero prob of R==1.0, C > R might fail to select anything.
    I = sum(I) || numel(C); % handle rare case where R==1.0
    
    X = Xs(I);
end

fprintf('Bayes Filter\n');
B = ones(NStates, 1) / NStates;
S = floor(rand(1) * NStates) + 1;
for PP = 1:10000
    A = floor(rand(1) * 4) + 1;
    [S2 Pr] = NormalStateTransitions(S, A);
    NS = sampleDiscrete(S2, Pr);
    Z = (S == NS);
    S = NS;
    B = bayesFilter(B, A, Z, NormalStateTransitions, @Observations);
    
    vizBel(3, B);
    
    if (max(B) == 1)
        break;
    end
end
fprintf('Bayes Filter: Iterations before belief convergence: %d\n', PP);

fprintf('QMDP\n');
function A = computeAction(B, Q)
    QQ = zeros(NActions, 1);
    for A = 1:NActions;
        QQ(A) = sum(B .* Q(:, A));
    end
    [~, A] = max(QQ);
end

% Recompute state transition table without making goal state absorbing.
QMDPStateTransitionTable = computeStateTransitionTable();
QMDPStateTransitions = @(S, A) deal(QMDPStateTransitionTable(S, A), 1);

% Value iteration won't converge if we don't make the goal state absorbing.
% We keep the value function computed with the absorbing goal state.

% TODO - only stop if belief has converged AND we are in goal state.
for SS = 1:NStates
    fprintf('QMDP: Start at state %d\n', SS);
    B = ones(NStates, 1) / NStates;
    S = SS;
    Path = S;
    for PP = 1:10000
        A = computeAction(B, Q);
        [S2 Pr] = QMDPStateTransitions(S, A);
        NS = sampleDiscrete(S2, Pr);
        Z = (S == NS);
        Path = [Path; NS];
        B = bayesFilter(B, A, Z, QMDPStateTransitions, @Observations);

        vizBel(5, B);
        PathX = [];
        PathY = [];
        for PI = 1:size(Path, 1)
            P = posFromState(Path(PI));
            
            PathX = [PathX; P(1)];
            PathY = [PathY; P(2)];
        end
        hold on;
        plot(PathX, PathY, 'b');
        pause(1);

        if (max(B) == 1) % TODO && S == GoalState
            break;
        end
        
        S = NS;
    end
    fprintf('QMDP: Start at state %d, iterations before belief convergence: %d\n', SS, PP);
end

end
