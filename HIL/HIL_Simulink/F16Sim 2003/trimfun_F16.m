function [cost, Xdot, xu] = trimfun_F16(UX0)
xu=UX0;
Xdot= feval('F16trim', 0, [], xu, 'outputs');
weight=[1 0.1 1 1 1 1];
cost = weight*(Xdot.*Xdot);
end