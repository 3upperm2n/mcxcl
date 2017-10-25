function f = myfun_zodiac(x)
% Objective function f()

wk_m = 100;   % simulation photons in million unit


% linear model for each device	
% zodiac: 0100	RX 480 (AMD)                     82 * x + 1021 = y (ms)
% zodiac: 0010	R9 Nano (AMD)                    75 * x + 986 = y 
% zodiac: 0001	Genuine Intel(R) CPU @ 2.00GHz   873.75 * x + 1172.25 = y  


f(1) = 82 * x(1) * wk_m + 1021;
f(2) = 75 * x(2) * wk_m  + 986;
f(3) = 873.75 * x(3) * wk_m  + 1172.25;

end