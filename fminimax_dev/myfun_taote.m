function f = myfun_taote(x)
% Objective function f()

wk_m = 100;   % simulation photons in million unit

% linear model for each device
% 1080ti : 58 * x + 89 = y (ms)
% 980 ti: 95.25 * x + 116.75 = y (ms)

f(1) = 58 * x(1) * wk_m + 89;
f(2) = 95.25 * x(2) * wk_m  + 116.75;

end