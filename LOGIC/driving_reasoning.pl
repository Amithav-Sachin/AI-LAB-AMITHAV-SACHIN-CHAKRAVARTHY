% ==============================================================================
% Laboratory: Logical Reasoning for Planning
% Task 8: Connect Prolog to Logical Reasoning
% ==============================================================================

% Ground Fact: The road is currently wet
wet_road.

% Rule 1: If the road is wet, it is slippery (wet_road -> slippery)
slippery :-
    wet_road.

% Rule 2: If the road is slippery, speed must be reduced (slippery -> reduce_speed)
reduce_speed :-
    slippery.
