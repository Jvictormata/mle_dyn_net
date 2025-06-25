clear all
clc

N = 500;

load("systems100.mat");

for i=1:100
i
theta = thetas(i,:);

sigma1 = theta(19);
sigma2 = theta(20);
sigma3 = theta(21);


e1 =  [(0:N-1)'  sigma1*randn(N,1)];
e2 =  [(0:N-1)'  sigma2*randn(N,1)];
e3 =  [(0:N-1)'  sigma3*randn(N,1)];


r1 =  [(0:N-1)' idinput([N 1],'rbs',[],[-1,1]) ];
r2 =  [(0:N-1)' idinput([N 1],'rbs',[],[-1,1]) ];
r3 =  [(0:N-1)' idinput([N 1],'rbs',[],[-1,1]) ];

out = sim('network_armax.slx',N-1);

y1 = out.y1;
y2 = out.y2;
y3 = out.y3;

r1 = out.r1;
r2 = out.r2;
r3 = out.r3;

u1 = y2+y3+r1;
u2 = r2;
u3 = y1+r3;

x = [y1(1:N);y2(1:N);y3(1:N);u1(1:N);u2(1:N);u3(1:N)];
r = [r1(1:N) r2(1:N) r3(1:N)];

xs(:,i) = x;
rs(:,:,i) = r;
end

%save("data_100_systems.mat","xs","rs")
save("validation_data_100_systems.mat","xs","rs")