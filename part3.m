%% Part III - QMDP

fprintf('QMDP\n');
function [As Pr] = computeAction(B, Q)
    QQ = zeros(NActions, 1);
    for LA = 1:NActions;
        QQ(LA) = sum(B .* Q(:, LA));
    end
    [Max, ~] = max(QQ);
    
    As = find(QQ == Max);
    Pr = ones(size(As, 1), 1) / size(As, 1);
end

% Recompute state transition table without making goal state absorbing.
QMDPStateTransitionTable = computeStateTransitionTable();
QMDPStateTransitions = @(S, A) deal(QMDPStateTransitionTable(S, A), 1);
QMDPObservations = @(S, A) Observations(S, A, QMDPStateTransitions);

% Value iteration won't converge if we don't make the goal state absorbing.
% We keep the value function computed with the absorbing goal state.
MaxIter = 100;
for SS = 1:NStates
    fprintf('QMDP: Start at state %d\n', SS);
    B = ones(NStates, 1) / NStates;
    S = SS;
    Path = S;
    for PP = 1:MaxIter
        [As Pr] = computeAction(B, Q);
        A = sampleDiscrete(As, Pr);
        [S2 Pr] = QMDPStateTransitions(S, A);
        NS = sampleDiscrete(S2, Pr);
        Z = (S == NS);
        Path = [Path; NS];
        B = bayesFilter(B, A, Z, QMDPStateTransitions, QMDPObservations);
        
        % pause(1);

        S = NS;
        
        % Stop if belief has converged and we are in goal state
        if (max(B) == 1 && S == GoalState)
            break;
        end
    end
    
    vizBel(5, B);
    hold on;
    PathX = [];
    PathY = [];
    for PI = 1:size(Path, 1)
        P = posFromState(Path(PI));
        PathX = [PathX; P(1)];
        PathY = [PathY; P(2)];
    end
    plot(PathX, PathY, 'b');

    if PP==MaxIter
        fprintf('QMDP: Start at state %d, did not converge\n', SS);
    else
        fprintf('QMDP: Start at state %d, iterations before belief convergence: %d\n', SS, PP);
    end
end
writeFigureEPS('QMDP-12.eps');
