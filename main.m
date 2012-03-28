%% Setup

clc;
clear all;
close all;

%% Design

% Firstly, since we are working in Matlab, we will not attempt to abstract
% over the details of the problem and produce a general algorithm, since
% the facilities available for doing so are relatively limited. Instead we
% will specify the problem and its representation and then write algorithms
% that solve the specific problem. That said, we will abstract a little
% where it is useful for code reuse between similar code for our
% algorithms.

% In some places we will make use of the fact that we have a tabular
% representation of a function, to speed up the algorithm; in others we
% will allow for a function to be either represented as a Matlab table or
% as a Matlab function, which will be either called or indexed with the
% same syntax, e.g. Q(S, A).


%% Part I - gridworld, dynamic programming

gridworld();

%% Part II - secretary problem, MC, TD (Q-learning)

secretary_MDP_MC();


