# Maximum Likelihood Estimation of Dynamical Networks


## Overview
Repository for the paper ``Identifiability and Maximum Likelihood Estimation for System Identification of Networks of Dynamical Systems''.

Note: Our goal by releasing the code is to make it reproducible, so that people can run it and be able to obtain the same results. It is not a library or package; the code is neither optimized nor designed for speed.

## Tutorial

Here is a short tutorial on how to use the functions implemented in "MLE_functions.py”

### Input information:
 * N = number of samples of each signal
 * M = number of internal systems
 * n_ab = number of parameters of each subsystem, n_ab = [[na1,na2,na3],[nb1,nb2,nb3]] for ARX models or n_ab = [[na1,na2,na3],[nb1,nb2,nb3],[nc1,nc2,nc3]] for the case where colored noise is present (ARMAX)
 * Lamb = interconnection matrix ($\Upsilon$ in the paper)
 * Delta = matrix of external signals ($\Omega$ in the paper)
 * r = vector with external signals
 * xo = column vector with the observed data
 * n_obs = number of observed signals
 * P = permutation matrix such that the observed signals are the leading entries of $\bar x = P^Tx $

 Note that x will be a column vector with dimension $2MN \times 1$.

 ##### Example: 
 
  - Network with 3 internal systems
  - 500 samples of each signal
  - First internal system: 1 zero, 2 poles, output connected to the internal system 3
  - Second internal system: 3 zeros, 4 poles, output connected to the internal system 1
  - Third internal system: 5 zeros, 6 poles, output connected to the internal system 1
  - Three external signals, r1 connected to the internal system 1, r2 connected to the internal system 2 and r3 connected to the internal system 3
  - Observed data: signals u1 and u3


 ###### Result:


  - N = 500
  - M = 3
  - n_ab = np.array([[2,4,5],[1,3,5]]) or n_ab = np.array([[2,4,5],[1,3,5],[2,4,5]]) for the ARMAX case
  - Lamb = np.array([[0,1,1],[0,0,0],[1,0,0]])
  - Delta = np.eye(3)
  - r = np.vstack((r1,r2,r3))
  - xo = np.vstack((u1,u3))
  - n_obs = 2
  - P = np.array([[0,0,1,0,0,0],[0,0,0,1,0,0],[0,0,0,0,1,0],[1,0,0,0,0,0],[0,0,0,0,0,1],[0,1,0,0,0,0]])

    Note that

$$
    x = \begin{bmatrix}
    y_1 \\\
    y_2 \\\
    y_3 \\\
    u_1 \\\
    u_2 \\\
    u_3
    \end{bmatrix}
$$

 and 


$$
    \bar{x} = \begin{bmatrix}
    u_1 \\\
    u_3 \\\
    y_1 \\\
    y_2 \\\
    y_3 \\\
    u_2
    \end{bmatrix}.
$$


### Generating the transformation matrices to obtain the non-singular pdf:
 - We first generate $A_2$ and $B_2$, using the function:

A2,B2 = gen_A2B2(Lamb,Delta,Permut,M,N,r_sig)

 - We then partition $A_2$ conformably with $\bar x^T = \[x_o^T \,\,
 , x_m^T\]$:

A2o = A2[:,:n_obs*N]

A2m = A2[:,n_obs*N:]

 - The transformation matrices are then given by:

 To_phi, Tm_phi, T_gamma, W2, V2 = get_transform_matrices(A2o,A2m,B2)

### Evaluating the MLE cost function, gradient, and hessian:
 For the case where the variances of all the external disturbances ($e^i$) are the same:

 * f = eval_cost_func(theta,n_ab,Permut,N,M,To_phi, Tm_phi, T_gamma, W2, xo)
 * g = grad_theta(theta.reshape(-1,1),n_ab,Permut,N,M,To_phi, Tm_phi, T_gamma, W2, xo)
 * H = Hessian_theta(theta,n_ab,Permut,N,M,To_phi, Tm_phi, T_gamma, W2, xo).reshape(theta.shape[0],theta.shape[0])

 For the general case where the variances of the external disturbances ($e^i$) are different:
 * f = eval_cost_func_dif_lambda(theta,n_ab,Permut,N,M,To_phi, Tm_phi, T_gamma, W2, xo)
 * g = grad_theta_dif_lambda(theta.reshape(-1,1),n_ab,Permut,N,M,To_phi, Tm_phi, T_gamma, W2, xo)
 * H = Hessian_theta_dif_lambda(theta,n_ab,Permut,N,M,To_phi, Tm_phi, T_gamma, W2, xo).reshape(theta.shape[0],theta.shape[0])
 
 The user can then implement the optimization method of his/her preference. Our numerical experiments used a trust region method implemented using the library optimize from Scipy. Notice that, since we are dealing with a non-convex optimization problem, an initialization strategy is required to ensure convergence to a good estimation.

 Our numerical experiments suggest that fitting the data first an ARX model, assuming all variances are equal, to initialize (a,b) and then estimating an ARMAX model assuming all variances are equal, to initialize (c,lambda), leads to a good initialization for (a,b,c,lambda) for the general problem.

 * The vector $\theta$ is defined 
 
$$   \theta = \begin{bmatrix}
    a_1 \\\
    b_1 \\\
    \vdots \\\
    a_M \\\
    b_M 
    \end{bmatrix}, 
$$

 for the ARX case with same lambdas

 $$ 
    \theta = \begin{bmatrix}
    a_1 \\\
    b_1 \\\
    c_1 \\\
    \vdots \\\
    a_M \\\
    b_M \\\
    c_M
    \end{bmatrix}, 
$$

for the ARMAX case with same lambdas, and

$$
    \quad \text{or} \quad 
    \theta = \begin{bmatrix}
    a_1 \\\
    b_1 \\\
    c_1 \\\
    \vdots \\\
    a_M \\\
    b_M \\\
    c_M \\\
    \log_{10}(\lambda_1^{-1/2}) \\\
    \vdots \\\
    \log_{10}(\lambda_M^{-1/2})
    \end{bmatrix},
$$

for the ARMAX case with different lambdas. Notice that we are scaling $\lambda_i$.


Check the code in "MLE - 1.1 Estimation (u1,u3) ARMAX - ARX init - 100 systems - N = 500 - TR.py" for a full example.


## Other useful functions

 *  P = permutation_mat(p)
 Given a list with the order of the variables it creates a permutation matrix, for example:
  p = [1,2,3,4] will lead to a permutation matrix equal to the identity $I_4$
  p = [2,1,3,4] will lead to a permutation matrix such that $\bar x = \begin(bmatrix)x_2\\ x_1 \\ x_3 \\ x_4\end(bmatrix)$.
  for the example of the network with three systems:
  p = [4,6,1,2,3,5] will lead to
  $$\bar x = \begin(bmatrix) x_4 \\ x_6 \\ x_1 \\ x_2 \\ x_3 \\ x_5 \end(bmatrix) =  \begin(bmatrix) u_1 \\ u_3 \\ y_1 \\ y_2 \\ y_3 \\ u_2 \end(bmatrix).$$ 


  * A1 = gen_A1(theta,n_ab,P,N)
  Generates $A_1$ as in (4).

  * a,b = get_ab(theta,n_ab)
  Returns the coefficients of the numerator and denominator of the transfer functions of each transfer function, a = list of denominators, b = list of numerators.




## Contact
If you have any questions or want to collaborate, contact me on LinkedIn
- https://www.linkedin.com/in/jvictormata/

---
