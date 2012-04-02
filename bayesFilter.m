function [NewBel] = bayesFilter(Bel, A, Z, StateTransitions, Observations)
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