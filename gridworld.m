function gridworld()

%% World Setup

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

MapWidth = 8;
MapHeight = 8;
MapSize = [MapWidth MapHeight];
NStates = prod(MapSize);
GoalState = NStates;

stateFromPos = @(P) (P(1) - 1) + MapWidth * (P(2) - 1) + 1;
posFromState = @(S) [mod((S - 1),MapWidth) + 1, floor((S - 1)/MapWidth) + 1];


%% Walls
% TODO merge H and V walls, split doesn't make sense for either rendering
% or state transition table calculation.

% Horizontal wall representation: Y, StartX, EndX
Walls_H = [
    0.5, 0.5, 8.5;
    1.5, 0.5, 1.5;
    1.5, 7.5, 8.5;
    5.5, 3.5, 5.5;
    6.5, 2.5, 6.5;
    8.5, 0.5, 8.5
];

% Vertical wall representation: X, StartY, EndY
Walls_V = [
    0.5, 0.5, 8.5;
    2.5, 4.5, 6.5;
    3.5, 3.5, 5.5;
    4.5, 2.5, 4.5;
    5.5, 3.5, 5.5;
    6.5, 4.5, 6.5;
    8.5, 0.5, 8.5
];

% Compute state transition matrix from walls

% [State x Action] -> State
StateTransitionTable = zeros([NStates, NActions]);

for A = 1:NActions
    for S = 1:NStates
        P = posFromState(S);
        
        X = P(1);
        Y = P(2);
        NP = Actions(A,:) + P;
        MoveOK = 1;
        
        for W = 1:size(Walls_H, 1)
            WS = [Walls_H(W, 2) Walls_H(W, 1)];
            WE = [Walls_H(W, 3) Walls_H(W, 1)];
            MoveOK = MoveOK & ~testSegmentSegment(P, NP, WS, WE);
        end
        
        for W = 1:size(Walls_V, 1)
            WS = [Walls_V(W, 1) Walls_V(W, 2)];
            WE = [Walls_V(W, 1) Walls_V(W, 3)];
            MoveOK = MoveOK & ~testSegmentSegment(P, NP, WS, WE);
        end
        
        if (MoveOK)
            StateTransitionTable(S, A) = stateFromPos(NP);
        else
            StateTransitionTable(S, A) = S;
        end
    end
end

StateTransitionTable(GoalState, :) = repmat(GoalState, [NActions 1]);

%% Visualisation

% Policy representation: [State] -> Action
% from the goal state
StartPolicy = [
    1 4 4 4 4 4 4 3 ...
    1 4 4 1 1 1 4 3 ...
    1 4 4 2 4 3 4 3 ...
    1 1 4 2 4 2 4 3 ...
    1 2 4 2 3 2 4 3 ...
    1 2 1 1 1 2 1 4 ...
    1 2 3 3 1 1 2 4 ...
    1 2 3 3 1 1 2 4
]';

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

function drawWalls()
    for i = 1:size(Walls_H, 1)
        Y      = Walls_H(i,1);
        StartX = Walls_H(i,2);
        EndX   = Walls_H(i,3);

        line([StartX EndX], [Y Y], 'Color', 'red');
    end
    for i = 1:size(Walls_V, 1)
        X      = Walls_V(i,1);
        StartY = Walls_V(i,2);
        EndY   = Walls_V(i,3);

        line([X X], [StartY EndY], 'Color', 'red');
    end
end

function drawGlyph(Glyph, P)
    % TODO ugh
    XS = squeeze([Glyph(1:2:end,1), Glyph(2:2:end,1)])';
    YS = squeeze([Glyph(1:2:end,2), Glyph(2:2:end,2)])';
    line(P(1) + XS, P(2) + YS, 'Color', 'green');
end

function drawStateTransitions()
    [NStates NActions] = size(StateTransitionTable);
    for A = 1:NActions
        for S = 1:NStates
            P = posFromState(S);
            if (StateTransitionTable(S, A) ~= S)
                % TODO put action index at end...
                drawGlyph(squeeze(Arrow(A,:,:,:)), P);
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
    [NStates] = size(Policy);
    
    for S = 1:NStates
        P = posFromState(S);
        X = P(1);
        Y = P(2);
        drawActionImpl(ActionGlyphs, X, Y, Policy(S));
    end
end

function vizPolicy(Fig, V, Policy)
     figure(Fig);
        imagesc(reshape(V, MapSize));
        colormap('gray');
        drawWalls();
        drawPolicy(Policy);
    axis([0.5, 8.5, 0.5, 8.5], 'xy', 'square');
end

figure(1);
    drawWalls();
    drawStateTransitions();
    drawPolicy(StartPolicy);
axis([0.5, 8.5, 0.5, 8.5], 'xy', 'square');

writeFigurePDF('StartingPolicy.pdf');

reward = @(S, A, S2) -1 * (S ~= GoalState);

%% We implement state transition probabilities PP(a, s, s') as a function
%% PP(s, a) -> [States, Probs]; which returns a list of possible next
%% states and their probabilities.

NormalStateTransitions = @(S, A) deal(StateTransitionTable(S, A), 1);

%% Policy evaluation
function [V] = evaluatePolicyStep(V, Policy, StateTransitions, Reward, Discount)
    for S = 1:NStates
        A = Policy(S);
        [S2 Pr] = StateTransitions(S, A);
        NV = 0;
        for I = 1:size(S2, 2)
            NV = NV + Pr(I) * (Reward(S, A, S2(I)) + Discount * V(S2(I)));
        end
        V(S) = NV;
    end
end

function [V] = evaluatePolicy(Policy, StateTransitions, Reward, Discount, MaxIterations)
    V = zeros([NStates, 1]);
    for Iteration = 1:MaxIterations
        OldV = V;
        V = evaluatePolicyStep(V, Policy, StateTransitions, Reward, Discount);
        MaxDelta = max(abs(V - OldV));

        if (MaxDelta < 0.001)
            break;
        end
    end
    fprintf('Iterations before value fn convergence: %d\n', Iteration);
end

function [Policy] = computeGreedyPolicy(V, StateTransitions, Reward, Discount)
    Policy = zeros([NStates, 1]);
    for S = 1:NStates
        Q = zeros([NActions, 1]);
        for A = 1:NActions;
            [S2 Pr] = StateTransitions(S, A);
            for I = 1:size(S2, 2)
                Q(A) = Q(A) + Pr(I) * (Reward(S, A, S2(I)) + Discount * V(S2(I)));
            end
        end
        [~, A] = max(Q);
        Policy(S) = A;
    end
end

fprintf('Policy evaluation\n');
Discount = 1;
MaxIterations = 1000;
V = evaluatePolicy(StartPolicy, NormalStateTransitions, reward, Discount, MaxIterations);
Policy = computeGreedyPolicy(V, NormalStateTransitions, reward, Discount);
vizPolicy(2, V, Policy);
writeFigurePDF('PolicyEvaluation.pdf');

%% Policy iteration

function policyIteration(Fig, Policy, Discount, StateTransitions, MaxValueIterations, MaxPolicyIterations)
    for PP = 1:MaxPolicyIterations
        % Evaluate policy
        V = evaluatePolicy(Policy, StateTransitions, reward, Discount, MaxValueIterations);

        % Compute greedy policy
        NewPolicy = computeGreedyPolicy(V, StateTransitions, reward, Discount);
        if (all(Policy == NewPolicy))
            break;
        end
        Policy = NewPolicy;

        vizPolicy(Fig, V, Policy);

        % refresh;
        % pause(0.1);
    end
    fprintf('Policy Iteration: Iterations before policy convergence: %d\n', PP);
end
fprintf('Policy iteration\n');
Discount = 1;
MaxValueIterations = 1000;
MaxPolicyIterations = 1000;
policyIteration(3, StartPolicy, Discount, NormalStateTransitions, MaxValueIterations, MaxPolicyIterations);
writeFigurePDF('NormalPolicyIteration.pdf');

%% Value iteration

function [NV] = valueIterationStep(V, StateTransitions, Reward, Discount)
    NV = zeros([NStates, 1]);
    for S = 1:NStates
        Q = zeros([NActions, 1]);
        for A = 1:NActions;
            [S2 Pr] = StateTransitions(S, A);
            for I = 1:size(S2, 2)
                Q(A) = Q(A) + Pr(I) * (Reward(S, A, S2(I)) + Discount * V(S2(I)));
            end
        end
        [V2 Dummy] = max(Q);
        NV(S) = V2;
    end
end

function valueIteration(Fig, Discount, StateTransitions, MaxPolicyIterations)
    V = zeros([NStates 1]);
    for PP = 1:MaxPolicyIterations
        % Evaluate policy

        % TODO why does 1-step evaluation not give the same result?
        % NewV = evaluatePolicy(Policy, NormalStateTransitions, reward, Discount, MaxIterations);
        NewV = valueIterationStep(V, StateTransitions, reward, Discount);

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
    fprintf('Value Iteration: Iterations before policy convergence: %d\n', PP);
end
fprintf('Value iteration\n');
Discount = 1;
MaxPolicyIterations = 1000;
valueIteration(4, Discount, NormalStateTransitions, MaxPolicyIterations);
writeFigurePDF('NormalValueIteration.pdf');

% Sticky wall
StickyProb = 0.4;

function [S2 Pr] = stickyStateTransitions(S, A)
    P = posFromState(S);
    X = P(1);
    if X == MapWidth
        S2 = [S, StateTransitionTable(S, A)];
        Pr = [StickyProb, 1 - StickyProb];
    else
        S2 = StateTransitionTable(S, A);
        Pr = 1;
    end
end

Discount = 1;
MaxValueIterations = 1000;
MaxPolicyIterations = 1000;
fprintf('Sticky world policy iteration\n');
policyIteration(5, StartPolicy, Discount, @stickyStateTransitions, MaxValueIterations, MaxPolicyIterations);
writeFigurePDF('StickyPolicyIteration.pdf')
fprintf('Sticky world value iteration\n');
valueIteration(6, Discount, @stickyStateTransitions, MaxPolicyIterations);
writeFigurePDF('StickyValueIteration.pdf');

StickyProb = 0.6;
fprintf('Sticky world policy iteration - p = 0.6\n');
policyIteration(7, StartPolicy, Discount, @stickyStateTransitions, MaxValueIterations, MaxPolicyIterations);
writeFigurePDF('StickyPolicyIteration6.pdf');
end
