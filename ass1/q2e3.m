% COMP9334 
% Week 4, Database server example
% After working out the balance equation, we need to solve
% the set of linear equations
% 
% I have chosen to use the first 5 equations on page 28 together
% with sum( probabilities ) = 1 
% 
% In principle, you can choose any of the 5 equations together
% with sum( probabilities ) = 1 
%
% We put the linear equations in standard form A x = b
% where x is the unknown vector  
% 
A = [ 15  -3   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
       0  15  -6   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
       0   0  15  -9   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
       0   0   0  15 -12   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
       0   0   0   0  15 -12   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
       0   0   0   0   0  15 -12   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
       0   0   0   0   0   0  15 -12   0   0   0   0   0   0   0   0   0   0   0   0   0   0
       0   0   0   0   0   0   0  15 -12   0   0   0   0   0   0   0   0   0   0   0   0   0
       0   0   0   0   0   0   0   0  15 -12   0   0   0   0   0   0   0   0   0   0   0   0
       0   0   0   0   0   0   0   0   0  15 -12   0   0   0   0   0   0   0   0   0   0   0
       0   0   0   0   0   0   0   0   0   0  15 -12   0   0   0   0   0   0   0   0   0   0
       0   0   0   0   0   0   0   0   0   0   0  15 -12   0   0   0   0   0   0   0   0   0
       0   0   0   0   0   0   0   0   0   0   0   0  15 -12   0   0   0   0   0   0   0   0
       0   0   0   0   0   0   0   0   0   0   0   0   0  15 -12   0   0   0   0   0   0   0
       0   0   0   0   0   0   0   0   0   0   0   0   0   0  15 -12   0   0   0   0   0   0
       0   0   0   0   0   0   0   0   0   0   0   0   0   0   0  15 -12   0   0   0   0   0
       0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0  15 -12   0   0   0   0
       0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0  15 -12   0   0   0
       0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0  15 -12   0   0
       0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0  15 -12   0
       0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0  15 -12
       1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1];
b = [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1]';
x = A\b