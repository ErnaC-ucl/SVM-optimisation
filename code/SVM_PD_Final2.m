clear all;clc;
data=csvread('data_train2.csv');
%z-score normalization
data(:,1:end-1)=zscore(data(:,1:end-1));
%%Initial Visualisation of data 
x = data(:,1:end-1);
y = data(:,end);

[N,M]=size(x);
%% Setting up the Primal Dual objective function
% Finding D
C=3; %penalty term for mis-classifications

D = zeros(length(x));
for i = 1: length(x)
    for j = 1:length(x)
        D(i,j) = y(i)*y(j)*x(i, 1:2)*x(j, 1:2)';
    end
end

eig(D)
% Objective function 
F.f = @(alpha_pd) 1/2*alpha_pd'*D*alpha_pd - alpha_pd'*ones(length(alpha_pd),1);
F.df = @(alpha_pd) D*alpha_pd - ones(length(alpha_pd),1);
F.d2f = @(alpha_pd) D;

% Inequality constraints
constraint2.f = @(alpha_pd) [alpha_pd-C*ones(length(alpha_pd),1); -alpha_pd];
constraint2.df = @(alpha_pd) [diag(ones(length(alpha_pd),1)'); diag(-ones(length(alpha_pd),1)')];
for i =1:2*N
    HessConst(:,:,i) = zeros(N, N);
end 
constraint2.d2f = @(alpha_pd) HessConst; 

% Structure to store inequality constraints
ineqConstraint = constraint2; %

% Equality constrain x^Ty = 0
eqConstraint.A = y';
eqConstraint.b = zeros(1,1);
% Structure to store equality constraints
eqConstraint = eqConstraint;



%% Parameters
%initialise alpha_pd0 for the dual optimisation problem so that it satisfies the equality constraint.
alpha_pd0_temp = 0.1*ones(length(x)-1,1); % choose n-1 parameters arbitrarily and compute the last one so that the equality constraint is satisfied
alpha_pd0_end=(-alpha_pd0_temp'*y(1:length(x)-1))/y(end)
alpha_pd0=[alpha_pd0_temp; alpha_pd0_end]
% -sum(x0(2:length(x0))'*y(2:length(x0)))/y(1)
lambda0 = ones(length(x)*2,1);% ensure: lambda0 > 0. This is the vector of inequality multipliers 
nu0 = ones(1,1); %scalar for equality multiplier
mu = 10; % in (3, 100);
t = 1; 
tol = 1e-12; %stopping criteria
tolFeas = 1e-12; %stopping criteria
maxIter = 500;

% Backtracking options
opts.maxIter = 30;
opts.alpha = 0.1; %1e-4; %0.1;
opts.beta = 0.8; %0.5;

%% Minimisation using the InteriorPoint_PrimalDual function
tic
[alpha_pd_k, fPD, tPD, nIter, infoPD] = interiorPoint_PrimalDual(F, ineqConstraint, eqConstraint, alpha_pd0, lambda0, nu0, mu, tol, tol, maxIter, opts);
% Assign values
alpha_pd_Min = alpha_pd_k;
fMin = F.f(alpha_pd_k);
toc
%% Iterate plot
iterations=1:nIter;
Fval=[]
for i=1:nIter
    input=infoPD.xs(:,i);
    Fval=[Fval,F.f(input)];
end 
figure;
hold on
plot(nIter,Fval,'b')
xlabel('Number of iterations')
ylabel('Value of the objective function')
hold off
%% Primal-Dual: residuals, surrogate gap
figure;
semilogy(infoPD.r_dual, 'LineWidth', 2); title('Primal-Dual')
hold on;
semilogy(infoPD.r_cent, 'LineWidth', 2);
semilogy(infoPD.r_prim, 'LineWidth', 2);
%semilogy(infoPD.r_prim + infoPD.r_dual + infoPD.r_cent, 'LineWidth', 2);
semilogy(infoPD.surgap, 'LineWidth', 2,  'Linestyle','--');
legend('Dual residual', 'Centering residual', 'Primal residual', 'Surrogate gap');
grid on;

%% Empirical convergence rates

% repeat for PD and barrier

conPD = convergenceHistory(infoPD, [], F, 2); % using 2-norm.
%conPD = convergenceHistory(infoPD, xBar, F, 2); % using 2-norm. (for C, PD struggles to reduce the dual residual, one of the eigevalues of KKT matrix close to 0/)


con = conPD; %{conPD, conBar}

% Q convergence: $||x_{k+1} - x^*|| / ||x_{k} - x^*||^p <= r$, convergence rate $r \in (0,1)$, convergence order $p$
p = 1; % convergance rate
rs1 = con.x(2:end)./(con.x(1:end-1).^p);
%p=2; rs2 = con.x(2:end)./(con.x(1:end-1).^p);
figure, plot(rs1, 'r'); hold on; grid on; title('Q convergence');
%hold on, plot(find(info.FRsteps(2:end)), rs1(logical(info.FRsteps(2:end))), 'xk', 'MarkerSize', 10);
%legend('p=1', 'FR step')
% 0 < r < 1 indicates linear convergence (after initial phase) for feasible x0
  
% Exponential convergence: ||x_{k} - x^*|| = c q^k, c>0, bounded, q in (0,1)
% ln ||x_{k} - x^*|| = ln c + (ln q)*k    [linear function in semilogy with slope (ln q)<0 and offset (ln c)]
% Note that exponential convergence is essentially Q convegence ||x_{k+1} - x^*|| / ||x_{k} - x^*||^p <= r for r=q and p=1
figure, semilogy(con.x, 'r'); grid on; title('Exponential convergence'); xlabel('k'); ylabel('ln(||x_{k} - x^*||')
if isfield(con,'x_nw') % Barrier
figure, semilogy(con.x_nw, 'r'); grid on; title('Exponential convergence (Newton iterates)'); xlabel('k'); ylabel('ln(||x_{k} - x^*||')
hold on, semilogy(find(infoBar.alphas_nw > 1-eps), con.x_nw(infoBar.alphas_nw > 1-eps), 'xk', 'MarkerSize', 10);
legend('p=1', 'Full step')
else %PD
hold on, semilogy(find(infoPD.s > 1-eps), con.x(infoPD.s > 1-eps), 'xk', 'MarkerSize', 10);
legend('p=1', 'Full step')
end
% Linear convergence after initial phase for feasible x0
  
% This is not a right way of looking at suspected linear convergence.
% Algebraic convergence: ||x_{k} - x^*|| = c k^{-p}, c>0, bounded
% ln ||x_{k} - x^*|| = ln c + -p*(ln k)     [linear function in log log with slope -p and offset (ln c)]
%figure, loglog(con.x, 'r'); grid on; title('Algebraic convergence'); xlabel('ln(k)'); ylabel('ln(||x_{k} - x^*||')

%% Visualization of Data
figure
hold on
scatter(x(y==1,1),x(y==1,2),'+g')
scatter(x(y==-1,1),x(y==-1,2),'.r')
xlabel('{x_1}')
ylabel('{x_2}')
legend('Positive Class','Negative Class','Location','southeast')
title('Data for classification')
hold off
    
%% Retriving betas
%https://uk.mathworks.com/matlabcentral/fileexchange/63158-support-vector-machine
Beta=zeros(M,1)';
alpha=alpha_pd_Min;
Xs=x(alpha>0,:); Ys=y(alpha>0);
Support_vectors=size(Xs,1);

Xs
Beta=(alpha(alpha>0).*Ys)'*Xs

% Solving for any support vector to obtain beta0
Beta_0=mean(Ys-(Xs*Beta'))

data_test=csvread('data_test.csv');

Xtest=data_test(:,1:end-1);
Ytest=data_test(:,end);
% f~ (Predicted labels)
f=sign(Xtest*Beta'+Beta_0);
%[F_measure, Accuracy] = confusion_mat(Ytest,f)
[~, Accuracy, F_measure ] = confusionMatrix(Ytest,f )

ft=x*Beta'+Beta_0;
zeta=max(0,1-y.*ft);
Non_Zero_Zeta=sum(zeta~=0);

X=x;
Y=y;

figure
hold on
scatter(X(Y==1,1),X(Y==1,2),'b')
scatter(X(Y==-1,1),X(Y==-1,2),'r')
scatter(Xs(Ys==1,1),Xs(Ys==1,2),'.b')
scatter(Xs(Ys==-1,1),Xs(Ys==-1,2),'.r')
syms x
fn=vpa((-Beta_0-Beta(1)*x)/Beta(2),4);
fplot(fn,'Linewidth',2);
fn1=vpa((1-Beta_0-Beta(1)*x)/Beta(2),4);
fplot(fn1,'--');
fn2=vpa((-1-Beta_0-Beta(1)*x)/Beta(2),4);
fplot(fn2,'--');
axis([-3 3 -3 3])
xlabel('{x_1}')
ylabel('{x_2}')
title('SVM Decision Boundary: PD Method')
legend('+ class','- class','support vector (+)','support vector (-)','Decision Boundry','Location','southeast')
hold off


