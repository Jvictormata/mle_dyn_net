clear all
clc

load("data_100_systems.mat")

N = 500;

load("systems100.mat");

theta = thetas(1,:);

sigma1 = theta(19);
sigma2 = theta(20);
sigma3 = theta(21);

r = rs(:,:,1);

r1 =  [(0:N-1)'  r(:,1)];
r2 =  [(0:N-1)'  r(:,2)];
r3 =   [(0:N-1)'  r(:,3)];


x_100 = [];
for i=1:100
    e1 =  [(0:N-1)'  sigma1*randn(N,1)];
    e2 =  [(0:N-1)'  sigma2*randn(N,1)];
    e3 =  [(0:N-1)'  sigma3*randn(N,1)];

out = sim('network_armax.slx',N-1);

y1 = out.y1;
y2 = out.y2;
y3 = out.y3;

u1 = y2+y3+out.r1;
u2 = out.r2;
u3 = y1+out.r3;


x = [y1(1:N);y2(1:N);y3(1:N);u1(1:N);u2(1:N);u3(1:N)];
x_100 = [x_100 x];
i
end

save("x_100_realizations.mat","x_100")