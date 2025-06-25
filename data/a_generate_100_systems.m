clear all
clc

M = 3;
N = 5000;

n_systems = 100;
lambda_max = 0.333;  % max amplitute for the noise sequences - notice that it makes lambda_bar = 0.1

L =  100;

e1 =  [(0:N-1)' linspace(1,1,N)' ];
e2 =  [(0:N-1)' linspace(1,1,N)' ];
e3 =  [(0:N-1)' linspace(1,1,N)' ];


r1 =  [(0:N-1)' linspace(1,1,N)' ];
r2 =  [(0:N-1)' linspace(1,1,N)' ];
r3 =  [(0:N-1)' linspace(1,1,N)' ];


trial = 0
thetas = zeros(n_systems,6*M+M);
i = 1
while i<= n_systems 
while true
    trial = trial +1
    
    [A1,B1,C1] = randARMAX_drss();
    [A2,B2,C2] = randARMAX_drss();
    [A3,B3,C3] = randARMAX_drss();
    theta = [A1(2:3)' B1' C1(2:3)' A2(2:3)' B2' C2(2:3)' A3(2:3)' B3' C3(2:3)'];

    rts = conv(A2',conv(A1',A3') - conv([0 B1'],[0 B3']));
    rts = max(abs(roots(rts)));

    out = sim('network_armax.slx',N-1);
    y1 = out.y1;
    y2 = out.y2;
    y3 = out.y3;

    n = size(y3,1);
if norm(y1(n))<L && norm(y2(n))<L && norm(y3(n))<L && rts < 0.95
    break
end
end

thetas(i,:) = [theta unifrnd(0.1*lambda_max, lambda_max) unifrnd(0.1*lambda_max, lambda_max) unifrnd(0.1*lambda_max, lambda_max)];
i = i+1
end

save("systems100.mat","thetas")







function [A,B,C] = randARMAX_drss()
% randARMAX_DRSS  Second-order ARMAX generator that relies on drss
% Uses:  Control System Toolbox  (drss, tfdata, tf)

% ------------------------------------------------------------------------
if nargin==0, Ts = 1; end                

% ----- make "nice" row vectors, drop leading zeros -------------
trim = @(v) v(find(abs(v)>eps,1,'first'):end).';

% ====================   A(z)  and B(z)  ===========================
while true
    % 2-state, 1-in, 1-out random *stable* discrete SS model
    sysA = drss(2,1,1);                 % pole radii < 1
    sysA.Ts = 1;                       % stamp sample time

    [numA,denA] = tfdata(sysA,'v');     % coefficient vectors (z⁻¹ domain)
    numA = trim(numA);  denA = trim(denA);

    poles = abs(roots(denA));

    % want deg(B)=1, deg(A)=2 and A monic and poles < 0.95 -------------------------------
    if numel(denA)==3 && numel(numA)==2 && max(poles)<0.95
        denA = denA/denA(1);            % → A(0)=1
        numA = numA/denA(1);            % same scaling for B
        break                           % good to go
    end
end
A = denA;      B = numA;                % store polynomials

% =======================    C(z)   ================================
while true
    sysC = drss(2,1,1);  sysC.Ts = 1;  % another independent drss draw
    [~,numC] = tfdata(sysC,'v');
    numC = trim(numC);
    
    zerosC = abs(roots(numC));

    if numel(numC)==3 &&  min(1-zerosC)>0.05     % need *exactly* order-2 numerator and c(z) not in the unit circle
        numC = numC/numC(1);             % make it monic
        break
    end
end
C = numC;
end
