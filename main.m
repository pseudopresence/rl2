%% Setup

clc;
clear all;
close all;
dbstop if error;
dbstop if infnan;

%% Part I - MDP

% -1 to each action, 0 if reach goal state
mdp();


%% Part II - Bayes filter

% bayes filter has p(z|x) but we bump on action, not position.
% One solution is to modify bayes filter to use p(z|u,x)
% Another is to augment state to be (x, y, bumped).

% for all s'
%  temp = sum_s { p(s'|u, s) p(s) }
%  p(s') = \eta p(z|s', u) temp

% where eta is a normalising constant
%% Part III - QMDP


