% simulation volume in million unit
phnVolume = 100;

% cores on each device
cuda_cores = [896, 896, 48];


% linear model for each device	
% zodiac: 0100	RX 480 (AMD)                     82 * x + 1021 = y (ms)
% zodiac: 0010	R9 Nano (AMD)                    75 * x + 986 = y 
% zodiac: 0001	Genuine Intel(R) CPU @ 2.00GHz   873.75 * x + 1172.25 = y  


coef_a = [82, 75, 873.75];
coef_b = [1021, 986, 1172.25];

workload_partion = iterative_pred(cuda_cores, coef_a, coef_b, phnVolume)