% simulation volume in million unit
phnVolume = 100;

% cores on each device
cuda_cores = [3584, 2816];


% linear model for each device
% 1080ti : 58 * x + 89 = y (ms)
% 980 ti: 95.25 * x + 116.75 = y (ms)
coef_a = [58, 95.25];
coef_b = [89, 116.75];

workload_partion = iterative_pred(cuda_cores, coef_a, coef_b, phnVolume)