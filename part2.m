%% Part II - Bayes filter

% bayes filter has p(z|x) but we bump on action, not position.
% One solution is to modify bayes filter to use p(z|u,x)
% Another is to augment state to be (x, y, bumped).

% for all s'
%  temp = sum_s { p(s'|u, s) p(s) }
%  p(s') = \eta p(z|s', u) temp

% where eta is a normalising constant

function [Z2 Pr] = Observations(S, A, StateTransitions)
    [S2 Pr] = StateTransitions(S, A);
    PS1 = sum(Pr(S2 == S));
    
    Z2 = [    0   1];
    Pr = [1-PS1 PS1];
end

NormalObservations = @(S, A) Observations(S, A, NormalStateTransitions);

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
    B = bayesFilter(B, A, Z, NormalStateTransitions, NormalObservations);
    
    vizBel(3, B);
    
    if (max(B) == 1)
        break;
    end
end
fprintf('Bayes Filter: Iterations before belief convergence: %d\n', PP);
writeFigureEPS('BayesFilter.eps');