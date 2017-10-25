% init
x0 = [0.1; 0.1; 0.1];

A = [];
b = [];

Aeq=[1,1,1];
beq = 1;
lb = [0;0;0];
ub = [1;1;1];
[x,fval] = fminimax(@myfun_zodiac,x0, A, b, Aeq, beq, lb, ub);

x
fval