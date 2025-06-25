clear all
clc


N = 50;
N_orig = 500;

load("data/validation_data_100_systems.mat");
load("results/thetas_pem_u1u3_armax_N_50.mat");
load("results/H23s_eq_pem_u1u3_armax_N_50.mat");

pyData = py.numpy.load("results/theta_opt_u1u3_arx_init_TR_N_50.npy");  
thetas_mle_u1u3 = double(pyData);
pyData = py.numpy.load("results/times_u1u3_arx_init_TR_N_50.npy");  
times_mle_u1u3 = double(pyData);


pyData = py.numpy.load("results/theta_opt_u3_arx_init_TR_N_50.npy");  
thetas_mle_u3 = double(pyData);
pyData = py.numpy.load("results/times_u3_arx_init_TR_N_50.npy");  
times_mle_u3 = double(pyData);



pred = true;  %true  %false

fits_xo_pem_u1u3_sim = [];
fits_xo_mle_u1u3_sim = [];

fits_xo_pem_u1u3_pred = [];
fits_xo_mle_u1u3_pred = [];

fits_xo_mle_u3_sim = [];
fits_xo_mle_u3_pred = [];

order_barH = [];

for i=1:100
i
%%%%% Getting the validation data:

r1 =  [(0:N-1)' rs(1:N,1,i) ];
r2 =  [(0:N-1)' rs(1:N,2,i) ];
r3 =  [(0:N-1)' rs(1:N,3,i) ];


%%%%%%%% Real system: %%%%%%%%
x = xs(:,i);

y1r = x(1:N);
y2r = x(N_orig+1:N_orig+N);
y3r = x(2*N_orig+1:2*N_orig+N);

u1r = x(3*N_orig+1:3*N_orig+N);
u2r = x(4*N_orig+1:4*N_orig+N);
u3r = x(5*N_orig+1:5*N_orig+N);

% Making e = 0 for the simulation of the estimated systems:

e1 =  [(0:N-1)'  0*randn(N,1)];
e2 =  [(0:N-1)'  0*randn(N,1)];
e3 =  [(0:N-1)'  0*randn(N,1)];


%%%%%%%% Estimated system PEM - x_o = u1,u3: %%%%%%%%
fit_xo_pem_u1u3_u1_4_sim = [];
fit_xo_pem_u1u3_u3_4_sim = [];
fit_xo_pem_u1u3_u1_4_pred = [];
fit_xo_pem_u1u3_u3_4_pred = [];

for j = 0:3
theta = thetas_pem_u1u3_armax(i,j*18+1:j*18+18); 

% Simulation Data:

out = sim('data/network_armax.slx',N-1);

y1 = out.y1;
y2 = out.y2;
y3 = out.y3;

u1 = y2+y3+out.r1;
u2 = out.r2;
u3 = y1+out.r3;

fit_xo_pem_u1u3_u1_4_sim = [fit_xo_pem_u1u3_u1_4_sim (1 - norm([u1]-[u1r])/norm([u1r]-mean([u1r])))];
fit_xo_pem_u1u3_u3_4_sim = [fit_xo_pem_u1u3_u3_4_sim (1 - norm([u3]-[u3r])/norm([u3r]-mean([u3r])))];


% Prediction Data:

if pred
a1 = [theta(1:2)];
b1 = [theta(3:4)];
c1 = [theta(5:6)]; 

a2 = [theta(7:8)];
b2 = [theta(9:10)];
c2 = [theta(11:12)];

a3 = [theta(13:14)];
b3 = [theta(15:16)];
c3 = [theta(17:18)];

H1 = tf([1 c1],[1 a1],1);
G1 = tf([b1],[1 a1],1);
G2 = tf([b2],[1 a2],1);
G3 = tf([b3],[1 a3],1);

H23_eq = H23s_eq(i,j+1);

u3_pem = lsim(H1^(-1),lsim(G1,u1r) + r3(:,2)) + lsim(1-H1^(-1),u3r);
u1_pem = lsim(H23_eq^(-1),lsim(G2,r2(:,2)) + lsim(G3,u3r) + r1(:,2)) + lsim(1-H23_eq^(-1),u1r);

fit_xo_pem_u1u3_u1_4_pred = [fit_xo_pem_u1u3_u1_4_pred  (1 - norm([u1_pem]-[u1r])/norm([u1r]-mean([u1r])))];
fit_xo_pem_u1u3_u3_4_pred =[fit_xo_pem_u1u3_u3_4_pred   (1 - norm([u3_pem]-[u3r])/norm([u3r]-mean([u3r])))];

end
end

%mean_fit_4 = mean([fit_xo_pem_u1u3_u1_4_sim; fit_xo_pem_u1u3_u3_4_sim; fit_xo_pem_u1u3_u1_4_pred; fit_xo_pem_u1u3_u3_4_pred],1); 
[~, max_fit] =  max(fit_xo_pem_u1u3_u1_4_pred);

fits_xo_pem_u1u3_sim = [fits_xo_pem_u1u3_sim;[fit_xo_pem_u1u3_u1_4_sim(max_fit) fit_xo_pem_u1u3_u3_4_sim(max_fit)]];
fits_xo_pem_u1u3_pred = [fits_xo_pem_u1u3_pred;[fit_xo_pem_u1u3_u1_4_pred(max_fit) fit_xo_pem_u1u3_u3_4_pred(max_fit)]];
order_barH = [order_barH; max_fit];

%%%%%%%% Estimated system MLE - x_o = u1,u3: %%%%%%%%
theta = thetas_mle_u1u3(i,:);

% Simulation Data:

out = sim('data/network_armax.slx',N-1);

y1 = out.y1;
y2 = out.y2;
y3 = out.y3;

u1 = y2+y3+out.r1;
u2 = out.r2;
u3 = y1+out.r3;

fit_xo_mle_u1u3_u1 = 1 - norm([u1]-[u1r])/norm([u1r]-mean([u1r]));
fit_xo_mle_u1u3_u3 = 1 - norm([u3]-[u3r])/norm([u3r]-mean([u3r]));

fits_xo_mle_u1u3_sim = [fits_xo_mle_u1u3_sim;[fit_xo_mle_u1u3_u1 fit_xo_mle_u1u3_u3]];


% Prediction Data:
if pred

a1 = [theta(1:2)];
b1 = [theta(3:4)];
c1 = [theta(5:6)]; 

a2 = [theta(7:8)];
b2 = [theta(9:10)];
c2 = [theta(11:12)];

a3 = [theta(13:14)];
b3 = [theta(15:16)];
c3 = [theta(17:18)];


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

[xHat, xiHat] = kalmanPredict([out.r1 out.r2 out.r3]',[u1r u3r]', Fc, Gr, K, Ho, Jro);

fit_xo_mle_u1u3_u1 = 1 - norm([xHat(1,:)']-[u1r])/norm([u1r]-mean([u1r]));
fit_xo_mle_u1u3_u3 = 1 - norm([xHat(2,:)']-[u3r])/norm([u3r]-mean([u3r]));

fits_xo_mle_u1u3_pred = [fits_xo_mle_u1u3_pred;[fit_xo_mle_u1u3_u1 fit_xo_mle_u1u3_u3]];
end


%%%%%%%% Estimated system MLE - x_o = u3: %%%%%%%%
theta = thetas_mle_u3(i,:);

% Simulation Data:

out = sim('data/network_armax.slx',N-1);

y1 = out.y1;
y2 = out.y2;
y3 = out.y3;

u1 = y2+y3+out.r1;
u2 = out.r2;
u3 = y1+out.r3;

fit_xo_mle_u3 = 1 - norm([u3]-[u3r])/norm([u3r]-mean([u3r]));
fits_xo_mle_u3_sim = [fits_xo_mle_u3_sim;fit_xo_mle_u3];


% Prediction Data:
if pred

a1 = [theta(1:2)];
b1 = [theta(3:4)];
c1 = [theta(5:6)]; 

a2 = [theta(7:8)];
b2 = [theta(9:10)];
c2 = [theta(11:12)];

a3 = [theta(13:14)];
b3 = [theta(15:16)];
c3 = [theta(17:18)];


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
if K
    [xHat, xiHat] = kalmanPredict([out.r1 out.r2 out.r3]',[u3r]', Fc, Gr, K, Ho, Jro);
    
    xo_pred_MLE = [xHat(1,:)'];
    fit_pred_MLE_u3 = 1 - norm(xo_pred_MLE-u3r)/norm(u3r - mean(u3r));
    
    fits_xo_mle_u3_pred = [fits_xo_mle_u3_pred;fit_pred_MLE_u3];
else
     fits_xo_mle_u3_pred = [fits_xo_mle_u3_pred;-10000];
end
end

end


converge_MLE_u1u3 = sum(fits_xo_mle_u1u3_sim>0.1)
converge_PEM_u1u3  = sum(fits_xo_pem_u1u3_sim>0.1)

temp1 = fits_xo_mle_u1u3_sim(:,1);
temp2 = fits_xo_mle_u1u3_sim(:,2);
fits_xo_mle_u1u3_sim_filtered = [temp1(temp1>0.1) temp2(temp1>0.1)];
temp3 = fits_xo_mle_u1u3_pred(:,1);
temp4 = fits_xo_mle_u1u3_pred(:,2);
fits_xo_mle_u1u3_pred_filtered = [temp3(temp1>0.1) temp4(temp1>0.1)];

temp1 = fits_xo_pem_u1u3_sim(:,1);
temp2 = fits_xo_pem_u1u3_sim(:,2);
fits_xo_pem_u1u3_sim_filtered = [temp1(temp1>0.1) temp2(temp1>0.1)];
temp3 = fits_xo_pem_u1u3_pred(:,1);
temp4 = fits_xo_pem_u1u3_pred(:,2);
fits_xo_pem_u1u3_pred_filtered = [temp3(temp1>0.1) temp4(temp1>0.1)];



mean_fit_xo_mle_sim_u1u3 = mean(fits_xo_mle_u1u3_sim_filtered)
mean_fit_xo_pem_sim_u1u3  = mean(fits_xo_pem_u1u3_sim_filtered)
std_fit_xo_mle_sim_u1u3 = std(fits_xo_mle_u1u3_sim_filtered)
std_fit_xo_pem_sim_u1u3  = std(fits_xo_pem_u1u3_sim_filtered)

mean_fit_xo_mle_pred_u1u3 = mean(fits_xo_mle_u1u3_pred_filtered)
mean_fit_xo_pem_pred_u1u3  = mean(fits_xo_pem_u1u3_pred_filtered)
std_fit_xo_mle_pred_u1u3 = std(fits_xo_mle_u1u3_pred_filtered)
std_fit_xo_pem_pred_u1u3  = std(fits_xo_pem_u1u3_pred_filtered)


converge_MLE_u3 = sum(fits_xo_mle_u3_sim>0.1)
mean_fit_xo_mle_sim_u3 = mean(fits_xo_mle_u3_sim(fits_xo_mle_u3_sim>0.1))
mean_fit_xo_mle_pred_u3  = mean(fits_xo_mle_u3_pred(fits_xo_mle_u3_pred>0.1))
std_fit_xo_mle_sim_u3 = std(fits_xo_mle_u3_sim(fits_xo_mle_u3_sim>0.1))
std_fit_xo_mle_pred_u3  = std(fits_xo_mle_u3_pred(fits_xo_mle_u3_pred>0.1))



avg_time_mle_u1u3 = mean(times_mle_u1u3)
avg_time_mle_u3 = mean(times_mle_u3)

[GC,GR] = groupcounts(order_barH)



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


