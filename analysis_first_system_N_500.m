clear all
clc

load("data/systems100.mat");
real_theta = thetas(1,1:18)';

N = 500;
N_orig = 500; 

%%%% MLE u1 u3
pyData = py.numpy.load("results/cov/theta_opt_u1u3_100_rel_arx_init_TR.npy"); 
thetas_mle_u1u3_100_rel = double(pyData);
thetas_mle_u1u3_100_rel_orig = thetas_mle_u1u3_100_rel';

thetas_mle_u1u3_100_rel = thetas_mle_u1u3_100_rel';
thetas_mle_u1u3_100_rel = thetas_mle_u1u3_100_rel(1:18,:);

%%%% MLE u3
pyData = py.numpy.load("results/cov/theta_opt_u3_100_rel_arx_init_TR.npy");  
thetas_mle_u3_100_rel = double(pyData);
thetas_mle_u3_100_rel_orig = thetas_mle_u3_100_rel';

thetas_mle_u3_100_rel = thetas_mle_u3_100_rel';
thetas_mle_u3_100_rel = thetas_mle_u3_100_rel(1:18,:);

% Considering only a,b:
real_theta(5:6) = 0;
real_theta(11:12) = 0;
real_theta(17:18) = 0;

thetas_mle_u1u3_100_rel(5:6,:) = 0;
thetas_mle_u1u3_100_rel(11:12,:) = 0;
thetas_mle_u1u3_100_rel(17:18,:) = 0;

thetas_mle_u3_100_rel(5:6,:) = 0;
thetas_mle_u3_100_rel(11:12,:) = 0;
thetas_mle_u3_100_rel(17:18,:) = 0;

% Loading data for fit:

load("data/validation_data_100_systems.mat");

x_val = xs(:,1);
r_val = rs(:,:,1);

y1r = x_val(1:N);
y2r = x_val(N_orig+1:N_orig+N);
y3r = x_val(2*N_orig+1:2*N_orig+N);

u1r = x_val(3*N_orig+1:3*N_orig+N);
u2r = x_val(4*N_orig+1:4*N_orig+N);
u3r = x_val(5*N_orig+1:5*N_orig+N);



%%
%%%%%%%% Computing FIT and Covariance - PEM - xo = (u1,u3): %%%%%%%%
load("data/data_100_systems.mat");
load("data/x_100_realizations.mat");

tolerance = 1e-5;

r = rs(:,:,1);

r1 = double(r(1:N,1));
r2 = double(r(1:N,2));
r3 = double(r(1:N,3));


thetas_pem_100 = [];
H23s_eq = []
for i =1:100

u1 = x_100(3*N_orig+1:3*N_orig+N,i);
u2 = x_100(4*N_orig+1:4*N_orig+N,i);
u3 = x_100(5*N_orig+1:5*N_orig+N,i);

Z1 = iddata(u3-r3,[u1]);
na = [2]; 
nb = [2];
nc = [2];
nk = [1];
opt = armaxOptions;
opt.SearchOptions.Tolerance = tolerance;
estG1H1 = armax(Z1, [na nb nc nk],opt)


Z2 = iddata(u1-r1,[u2 u3]);
nf = [2 2]; 
nb = [2 2];
nc = [2];  % 2 was the order for barH which led to the best fit values for prediction on the validation data for system 1 and N=500
nd = [2];  % 2 was the order for barH which led to the best fit values for prediction on the validation data for system 1 and N=500
nk = [1 1];
opt = bjOptions;
opt.SearchOptions.Tolerance = tolerance;
estG2G3 = bj(Z2, [nb nc nd nf nk],opt)

H23_eq = [estG2G3.C estG2G3.D]
H23s_eq = [H23s_eq; H23_eq];

theta_pem_u1u3_armax = [estG1H1.A(2:3) estG1H1.B(2:3) estG1H1.C(2:3) estG2G3.F{1}(2:3) estG2G3.B{1}(2:3) 0 0 estG2G3.F{2}(2:3) estG2G3.B{2}(2:3) 0 0];
thetas_pem_100 = [thetas_pem_100; theta_pem_u1u3_armax];
end


thetas_pem_100 = thetas_pem_100';
thetas_pem_100_orig = thetas_pem_100;
thetas_pem_100(5:6,:) = 0;
exp_theta_pem_u1u3 = mean(thetas_pem_100,2);

cov_pem_u1u3 = zeros(size(exp_theta_pem_u1u3,1),size(exp_theta_pem_u1u3,1));

difference = thetas_pem_100 - exp_theta_pem_u1u3;
for i=1:100
    cov_pem_u1u3 = cov_pem_u1u3 + difference(:,i)*(difference(:,i)');
end

cov_pem_u1u3 = cov_pem_u1u3/100;
result_cov_pem = [trace(cov_pem_u1u3),max(eig(cov_pem_u1u3))]

bias_pem_u1u3 =  norm(exp_theta_pem_u1u3 - real_theta)^2

mse_pem_u1u3 = trace(cov_pem_u1u3) + norm(exp_theta_pem_u1u3 - real_theta)^2


% Fit Simulation Data:

theta = mean(thetas_pem_100_orig,2);

r1 =  [(0:N-1)' r_val(1:N,1) ];
r2 =  [(0:N-1)' r_val(1:N,2) ];
r3 =  [(0:N-1)' r_val(1:N,3) ];

e1 =  [(0:N-1)'  0*randn(N,1)];
e2 =  [(0:N-1)'  0*randn(N,1)];
e3 =  [(0:N-1)'  0*randn(N,1)];

out = sim('data/network_armax.slx',N-1);

y1 = out.y1;
y2 = out.y2;
y3 = out.y3;

u1 = y2+y3+out.r1;
u2 = out.r2;
u3 = y1+out.r3;

fit_xo_pem_u1u3_u1 = 1 - norm([u1]-[u1r])/norm([u1r]-mean([u1r]));
fit_xo_pem_u1u3_u3 = 1 - norm([u3]-[u3r])/norm([u3r]-mean([u3r]));

fits_xo_pem_u1u3_sim = [fit_xo_pem_u1u3_u1 fit_xo_pem_u1u3_u3]

% Fit Prediction Data:

exp_H23s = mean(H23s_eq,1);
H23_eq = tf(exp_H23s(1:size(exp_H23s,2)/2),exp_H23s(size(exp_H23s,2)/2+1:size(exp_H23s,2)),1);

a1 = [theta(1:2)]';
b1 = [theta(3:4)]';
c1 = [theta(5:6)]'; 

a2 = [theta(7:8)]';
b2 = [theta(9:10)]';

a3 = [theta(13:14)]';
b3 = [theta(15:16)]';

H1 = tf([1 c1],[1 a1],1);
G1 = tf([b1],[1 a1],1);
G2 = tf([b2],[1 a2],1);
G3 = tf([b3],[1 a3],1);

u3_pem = lsim(H1^(-1),lsim(G1,u1r) + r3(:,2)) + lsim(1-H1^(-1),u3r);
u1_pem = lsim(H23_eq^(-1),lsim(G2,r2(:,2)) + lsim(G3,u3r) + r1(:,2)) + lsim(1-H23_eq^(-1),u1r);

fit_xo_pem_u1u3_u1 = 1 - norm([u1_pem]-[u1r])/norm([u1r]-mean([u1r]));
fit_xo_pem_u1u3_u3 = 1 - norm([u3_pem]-[u3r])/norm([u3r]-mean([u3r]));

fits_xo_pem_u1u3_pred = [fit_xo_pem_u1u3_u1 fit_xo_pem_u1u3_u3]



%%%%%%%% Computing FIT and Covariance - MLE - xo = (u1,u3): %%%%%%%%

% Cov:
exp_theta_mle_u1u3 = mean(thetas_mle_u1u3_100_rel,2);

difference = thetas_mle_u1u3_100_rel-exp_theta_mle_u1u3;

cov_mle_u1u3 = zeros(size(thetas_mle_u1u3_100_rel,1),size(thetas_mle_u1u3_100_rel,1));

for i=1:size(thetas_mle_u1u3_100_rel,2)
    cov_mle_u1u3 = cov_mle_u1u3 + difference(:,i)*(difference(:,i)');
end

cov_mle_u1u3 = cov_mle_u1u3/size(thetas_mle_u1u3_100_rel,2);
result_cov_mle_u1u3 =  [trace(cov_mle_u1u3) max(eig(cov_mle_u1u3))]

bias_mle_u1u3  = norm(exp_theta_mle_u1u3 - real_theta)^2
mse_mle_u1u3  = norm(exp_theta_mle_u1u3 - real_theta)^2 + trace(cov_mle_u1u3)


% Fit Simulation Data:
theta = mean(thetas_mle_u1u3_100_rel_orig,2);

r1 =  [(0:N-1)' r_val(1:N,1) ];
r2 =  [(0:N-1)' r_val(1:N,2) ];
r3 =  [(0:N-1)' r_val(1:N,3) ];

e1 =  [(0:N-1)'  0*randn(N,1)];
e2 =  [(0:N-1)'  0*randn(N,1)];
e3 =  [(0:N-1)'  0*randn(N,1)];


out = sim('data/network_armax.slx',N-1);

y1 = out.y1;
y2 = out.y2;
y3 = out.y3;

u1 = y2+y3+out.r1;
u2 = out.r2;
u3 = y1+out.r3;

fit_xo_mle_u1u3_u1 = 1 - norm([u1]-[u1r])/norm([u1r]-mean([u1r]));
fit_xo_mle_u1u3_u3 = 1 - norm([u3]-[u3r])/norm([u3r]-mean([u3r]));

fits_xo_mle_u1u3_sim = [fit_xo_mle_u1u3_u1 fit_xo_mle_u1u3_u3]

% Fit Prediction Data:

a1 = [theta(1:2)]';
b1 = [theta(3:4)]';
c1 = [theta(5:6)]'; 

a2 = [theta(7:8)]';
b2 = [theta(9:10)]';
c2 = [theta(11:12)]';

a3 = [theta(13:14)]';
b3 = [theta(15:16)]';
c3 = [theta(17:18)]';


U = zeros(2,2); U(1,2) = 1;


% F,B,C,H matrices:
F1 = U - a1'*[1 0]; 
F2 = U - a2'*[1 0];  
F3 = U - a3'*[1 0];  

B1 = b1'; 
B2 = b2'; 
B3 = b3'; 

F= blkdiag(F1,F2,F3);
B = blkdiag(B1,B2,B3);
C = blkdiag(c1'-a1',c2'-a2',c3'-a3');
H = blkdiag([1 0],[1 0],[1 0]);

% Closed loop matrices:
Ups = [0 1 1; 0 0 0; 1 0 0];
Ome = eye(3);


Fc  = F+B*Ups*H;
Gr = B*Ome;
Ge = C + B*Ups;
Hc = [eye(3); Ups]*H;
Jr =  [zeros(3,3); Ome];
Je =  [eye(3); Ups];

% Partial observation:
To = [0 0 0 1 0 0; 0 0 0 0 0 1];


Ho = To*Hc;
Jro = To*Jr;
Jeo = To*Je;

% Riccati equation: 
lambs = theta(19:21);
Sig_e = diag((1./10.^(lambs)).^2);

Q = Ge*Sig_e*Ge';
R = Jeo*Sig_e*Jeo';
S = Ge*Sig_e*Jeo';

[Sig,K,~] = idare(Fc',Ho',Q,R,S);
K = K';

[xHat, xiHat] = kalmanPredict([r_val(1:N,1) r_val(1:N,2) r_val(1:N,3)]',[u1r u3r]', Fc, Gr, K, Ho, Jro);

fit_xo_mle_u1u3_u1 = 1 - norm([xHat(1,:)']-[u1r])/norm([u1r]-mean([u1r]));
fit_xo_mle_u1u3_u3 = 1 - norm([xHat(2,:)']-[u3r])/norm([u3r]-mean([u3r]));

fits_xo_mle_u1u3_pred = [fit_xo_mle_u1u3_u1 fit_xo_mle_u1u3_u3]

%%
%%%%%%%% Computing FIT and Covariance - MLE - xo = u3: %%%%%%%%

exp_theta_mle_u3 = mean(thetas_mle_u3_100_rel,2);

difference = thetas_mle_u3_100_rel-exp_theta_mle_u3;

cov_mle_u3 = zeros(size(thetas_mle_u3_100_rel,1),size(thetas_mle_u3_100_rel,1));

for i=1:size(thetas_mle_u3_100_rel,2)
    cov_mle_u3 = cov_mle_u3 + difference(:,i)*(difference(:,i)');
end

cov_mle_u3 = cov_mle_u3/size(thetas_mle_u3_100_rel,2);
result_cov_mle_u3 =  [trace(cov_mle_u3) max(eig(cov_mle_u3))]

bias_mle_u3  = norm(exp_theta_mle_u3 - real_theta)^2
mse_mle_u3  = norm(exp_theta_mle_u3 - real_theta)^2 + trace(cov_mle_u3)


% Fit Simulation Data:
theta = mean(thetas_mle_u3_100_rel_orig,2);

r1 =  [(0:N-1)' r_val(1:N,1) ];
r2 =  [(0:N-1)' r_val(1:N,2) ];
r3 =  [(0:N-1)' r_val(1:N,3) ];

e1 =  [(0:N-1)'  0*randn(N,1)];
e2 =  [(0:N-1)'  0*randn(N,1)];
e3 =  [(0:N-1)'  0*randn(N,1)];


out = sim('data/network_armax.slx',N-1);

y1 = out.y1;
y2 = out.y2;
y3 = out.y3;

u1 = y2+y3+out.r1;
u2 = out.r2;
u3 = y1+out.r3;

fit_xo_mle_u3_sim = 1 - norm([u3]-[u3r])/norm([u3r]-mean([u3r]))


% Fit Prediction Data:

a1 = [theta(1:2)]';
b1 = [theta(3:4)]';
c1 = [theta(5:6)]'; 

a2 = [theta(7:8)]';
b2 = [theta(9:10)]';
c2 = [theta(11:12)]';

a3 = [theta(13:14)]';
b3 = [theta(15:16)]';
c3 = [theta(17:18)]';


U = zeros(2,2); U(1,2) = 1;


% F,B,C,H matrices:
F1 = U - a1'*[1 0]; 
F2 = U - a2'*[1 0]; 
F3 = U - a3'*[1 0]; 

B1 = b1'; 
B2 = b2'; 
B3 = b3'; 

F= blkdiag(F1,F2,F3);
B = blkdiag(B1,B2,B3);
C = blkdiag(c1'-a1',c2'-a2',c3'-a3');
H = blkdiag([1 0],[1 0],[1 0]);

% Closed loop matrices:
Ups = [0 1 1; 0 0 0; 1 0 0];
Ome = eye(3);


Fc  = F+B*Ups*H;
Gr = B*Ome;
Ge = C + B*Ups;
Hc = [eye(3); Ups]*H;
Jr =  [zeros(3,3); Ome];
Je =  [eye(3); Ups];

% Partial observation:
To = [0 0 0 0 0 1];

Ho = To*Hc;
Jro = To*Jr;
Jeo = To*Je;

% Riccati equation: 
lambs = theta(19:21);
Sig_e = diag((1./10.^(lambs)).^2);

Q = Ge*Sig_e*Ge';
R = Jeo*Sig_e*Jeo';
S = Ge*Sig_e*Jeo';

[Sig,K,~] = idare(Fc',Ho',Q,R,S);
K = K';

[xHat, xiHat] = kalmanPredict([r_val(1:N,1) r_val(1:N,2) r_val(1:N,3)]',[u3r]', Fc, Gr, K, Ho, Jro);

fit_xo_mle_u3_pred = 1 - norm([xHat']-[u3r])/norm([u3r]-mean([u3r]))



function [xHat, xiHat] = kalmanPredict(r, x_o, F_c, G_r, K, H_o, J_ro, xi0)
%KALMANPREDICT  Run a time‑invariant Kalman predictor over a data batch.
%
%   [xHat, xiHat] = kalmanPredict(r, x_o, F_c, G_r, K, H_o, J_ro, xi0)
%
%   INPUTS
%     r    : m×N   reference/disturbance sequence           (columns = k = 1…N)
%     x_o  : p×N   measured outputs                         (columns = k)
%     F_c  : n×n   state transition matrix
%     G_r  : n×m   input (reference) matrix
%     K    : n×p   steady‑state Kalman gain
%     H_o  : p×n   output matrix
%     J_ro : p×m   direct‑through matrix from r to x_o
%     xi0  : n×1   (optional) initial state estimate; default = zeros(n,1)
%
%   OUTPUTS
%     xHat  : p×N   sequence of predicted outputs  \hat x_{o,k}
%     xiHat : n×(N+1) state estimates,  \hat ξ_k   (xiHat(:,1) = xi0)
%
%   The predictor obeys
%       xî_{k+1} = F_c·xî_k + G_r·r_k + K·(x_{o,k} – H_o·xî_k – J_ro·r_k)
%       x̂_{o,k}  = H_o·xî_k + J_ro·r_k
%

% -------- sanity checks & defaults --------------------------------------
if nargin < 8 || isempty(xi0)
    xi0 = zeros(size(F_c,1),1);
end

[n, N] = size(x_o); %#ok<ASGLU>  % p = size(x_o,1); N timesteps
assert(size(r,2)==N,  'r and x_o must have the same number of columns/time‑steps');
assert(size(H_o,2)==size(F_c,1), 'Dim mismatch: H_o vs F_c');

% -------- pre‑allocate ---------------------------------------------------
p       = size(H_o,1);
xiHat   = zeros(size(F_c,1), N+1);
xHat    = zeros(p, N);

xiHat(:,1) = xi0;

% -------- main loop ------------------------------------------------------
for k = 1:N
    % measurement‑update (predict output)
    xHat(:,k)   = H_o*xiHat(:,k) + J_ro*r(:,k);

    % innovation
    innov       = x_o(:,k) - xHat(:,k);

    % time‑update (predict next state)
    xiHat(:,k+1) = F_c*xiHat(:,k) + G_r*r(:,k) + K*innov;
end
end


