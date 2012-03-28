function [] = writeFigureEPS(Fig, FileNameEPS)
% writeFigureEPS writes the current figure to an eps file
% INPUT Fig: [optional] the figure to write; default is the current figure.
%       FileName: a string containing the path of the file to save to.

if nargin < 2
    FileNameEPS = Fig;
    Fig = gcf;
end
SavedBreakpoints = dbstatus;
dbclear if infnan; % Yes, this is a thing
print(Fig, '-depsc', FileNameEPS);
dbstop(SavedBreakpoints);

