% ==============================================================================
% Laboratory: Logical Reasoning for Planning
% Task 6 & Task 7: Prolog as a Logical Verifier
% ==============================================================================

% --- Task 6: Warehouse Topology Facts ---
% The warehouse contains locations a, b, and c.
% Physical connectivity: a is connected to b, b is connected to c.
% Note: a and c are NOT directly connected.
connected(a, b).
connected(b, a).
connected(b, c).
connected(c, b).

% --- Task 6: Can Move Rule ---
% A robot can move from X to Y if X and Y are connected.
can_move(X, Y) :-
    connected(X, Y).

% --- Task 7: Valid Move Rule ---
% Extends check for valid single movements.
valid_move(X, Y) :-
    connected(X, Y).

% --- Extension: Multi-step Plan Verification Rule ---
% Verifies whether a sequence of move actions forms an unbroken valid trajectory.
valid_move_plan([]).
valid_move_plan([move(X, Y)]) :-
    valid_move(X, Y).
valid_move_plan([move(X, Y), move(Y, Z) | Rest]) :-
    valid_move(X, Y),
    valid_move_plan([move(Y, Z) | Rest]).
