clear all
clc


N = 500;       

tolerance = 1e-5;

load('data/data_100_systems.mat');
thetas_pem_u1u3_armax = [];
H23s_eq = [];

tic
for i=1:100

x = xs(:,i);
r = rs(:,:,i);

u1 = x(3*N+1:4*N);
u2 = x(4*N+1:5*N);
u3 = x(5*N+1:6*N);
r1 = double(r(:,1));
r2 = double(r(:,2));
r3 = double(r(:,3));

%%%%%%%%%%%       X_o = (u_1, u_3)        %%%%%%%%%%%%%%%


Z1 = iddata(u3-r3,[u1]);
na = [2]; 
nb = [2];
nc = [2];
nk = [1];
opt = armaxOptions;
opt.SearchOptions.Tolerance = tolerance;
estG1H1 = armax(Z1, [na nb nc nk],opt)

theta_pem_u1u3_armax_4 = [];
H23_eq_4 = []
for j=1:4
Z2 = iddata(u1-r1,[u2 u3]);
nf = [2 2]; 
nb = [2 2];
nc = [j]; 
nd = [j]; 
nk = [1 1];
opt = bjOptions;
opt.SearchOptions.Tolerance = tolerance;
estG2G3 = bj(Z2, [nb nc nd nf nk],opt)

H23_eq = tf(estG2G3.C,estG2G3.D,1);
theta_pem_u1u3_armax = [estG1H1.A(2:3) estG1H1.B(2:3) estG1H1.C(2:3) estG2G3.F{1}(2:3) estG2G3.B{1}(2:3) 0 0 estG2G3.F{2}(2:3) estG2G3.B{2}(2:3) 0 0];
H23_eq_4 = [H23_eq_4 H23_eq];
theta_pem_u1u3_armax_4 = [theta_pem_u1u3_armax_4 theta_pem_u1u3_armax];
end

thetas_pem_u1u3_armax = [thetas_pem_u1u3_armax; theta_pem_u1u3_armax_4];
H23s_eq = [H23s_eq; H23_eq_4];
end
total_time = toc;
average_time = total_time/100

save("results/thetas_pem_u1u3_armax.mat","thetas_pem_u1u3_armax")
save("results/H23s_eq_pem_u1u3_armax.mat","H23s_eq")


