function [NewBel] = bayesFilter(Bel, A, Z, StateTransitions, Observations)
% bayesFilter updates the current belief based on an action,
% an observation, the state transition probabilities, and observation
% probabilities.
% INPUT Bel: [NStates x 1]
%            Current belief as a dense discrete distribution.
%       A: [1x1]
%          Action taken.
%       Z: [1x1]
%          Observation.
%       StateTransitions: (State, Action) -> (States [Nx1], Probs [Nx1])
%          Function taking a state action pair, returning sparse
%          distribution of new states.
%       Observations: (State, Action) -> (Obs [Nx1], Probs [Nx1])
%          Function taking a state action pair, returning sparse
%          distribution of observations.
    NStates = size(Bel, 1);
    NewBel = zeros(NStates, 1);
    for NS = 1:NStates
        Temp = 0;
        for S = 1:NStates
            [S2 Pr] = StateTransitions(S, A);
            PS2 = sum(Pr(S2 == NS));
            
            [Z2 Pr] = Observations(S, A);
            PZ = sum(Pr(Z2 == Z));
            
            Temp = Temp + PZ * PS2 * Bel(S);
        end
        NewBel(NS) = Temp;
    end
    NewBel = NewBel / sum(NewBel);
end