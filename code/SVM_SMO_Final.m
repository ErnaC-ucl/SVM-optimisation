clear all;clc;
data=csvread('data_train2.csv');
%z-score normalization
data(:,1:end-1)=zscore(data(:,1:end-1));
x = data(:,1:end-1);
y = data(:,end);

%%Parameters

n = length(data); %Total number of data points
C = 0.5; %Regularization parameter, C (optimal) using Cross-Validation techniques
tol = 10e-5; %Tolerence
%Initializing Lagrange Multipliers alpha to the same alpha_0 as the PD method
alpha_temp = 0.1*ones(length(x)-1,1); % choose n-1 parameters arbitrarily and compute the last one so that the equality constraint is satisfied
alpha_end=(-alpha_temp'*y(1:length(x)-1))/y(end);
alpha=[alpha_temp; alpha_end];
bias = 0; %Initialise bias
info.xs = [];
nIter=0;
%% Setting up the SMO Algorithm for SVM optimisation
tic
while (1)
    
    changed_alphas=0;
    N=size(y,1);
    fprintf('Possible support vectors: %d',N)
    
    for i=1:N
        Inputs1=x*x(i,:)';
        Ei=sum(alpha.*y.*Inputs1)-y(i);
        
        if ((Ei*y(i) < -tol) && alpha(i) < C) || (Ei*y(i) > tol && (alpha(i) > 0))
            
            for j=[1:i-1,i+1:N]
                Inputs2=x*x(j,:)';
                Ej=sum(alpha.*y.*Inputs2)-y(j);
                  alpha_iold=alpha(i);
                  alpha_jold=alpha(j);
                  if y(i)~=y(j)
                      L=max(0,alpha(j)-alpha(i));
                      H=min(C,C+alpha(j)-alpha(i));
                  else 
                      L=max(0,alpha(i)+alpha(j)-C);
                      H=min(C,alpha(i)+alpha(j));
                  end
                  if (L==H)
                      continue
                  end
                  Dotxixj=x(j,:)*x(i,:)';
                  Dotxixi=x(i,:)*x(i,:)';
                  Dotxjxj=x(j,:)*x(j,:)';
                  eta = 2*Dotxixj-Dotxixi-Dotxjxj;
                  
                  if eta>=0
                      continue
                  end
                  
                  alpha(j)=alpha(j)-( y(j)*(Ei-Ej) )/eta;
                  
                  if alpha(j) > H
                      alpha(j) = H;
                  elseif alpha(j) < L
                      alpha(j) = L;
                  end
                  if norm(alpha(j)-alpha_jold) < tol
                      continue
                  end
                  
                  alpha(i)=alpha(i) + y(i)*y(j)*(alpha_jold-alpha(j));
                  b1 = bias - Ei - y(i)*(alpha(i)-alpha_iold)*Dotxixi...
                      -y(j)*(alpha(j)-alpha_jold)*Dotxixj;
                  
                  b2 = bias - Ej - y(i)*(alpha(i)-alpha_iold)*Dotxixj...
                      -y(j)*(alpha(j)-alpha_jold)*Dotxjxj;
           
                  if 0<alpha(i)<C
                      bias=b1;
                  elseif 0<alpha(j)<C
                      bias=b2;
                  else
                      bias=(b1+b2)/2;
                  end
                  
                  changed_alphas=changed_alphas+1;
            end
        end
    end
    
    if changed_alphas==0
        break
    end
    info.xs = [info.xs alpha];
    nIter=nIter+1;
end
nIter
toc
    
%% Retriving betas

%https://uk.mathworks.com/matlabcentral/fileexchange/63100-smo-sequential-minimal-optimization?s_tid=mwa_osa_a
%weights
alpha=info.xs(:,end)
Beta=sum(alpha.*y.*x)
%bias
Beta_0 =mean( y - x*Beta')
%Support vectors 
Xs=x(alpha>0,:); Ys=y(alpha>0);
fprintf('Number of support Vectors : %d',size(Xs,1))

%Accuracy and F-measure 
data_test=csvread('data_test.csv');
Xtest=data_test(:,1:end-1);
Ytest=data_test(:,end);
% f~ (Predicted labels)
f=sign(Xtest*Beta'+Beta_0);
%[F_measure, Accuracy] = confusion_mat(Ytest,f)
[~, Accuracy, F_measure ] = confusionMatrix(Ytest,f )

X=x; Y=y;
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
title('SVM Decision Boundary: SMO Method')
legend('+ class','- class','support vector (+)','support vector (-)','Decision Boundry','Location','southeast')
hold off


%% Objective function
info.xs
xMin=info.xs(:,end);
x = data(:,1:end-1);
y = data(:,end);
D = zeros(length(x));
for i = 1: length(x)
    for j = 1:length(x)
        D(i,j) = y(i)*y(j)*x(i,:)*x(j,:)';
    end
end
% Objective function 
F.f = @(x) 1/2*x'*D*x - x'*ones(length(x),1);
F.df = @(x) D*x - ones(length(x),1);
F.d2f = @(x) D;
fMin = F.f(xMin);

eig(D); %to verify if assumption of positive-definitiness is met for linear convergence 
info.xs;
iterations=1:nIter;
Fval=[];
for i=1:nIter
    input=info.xs(:,i);
    Fval=[Fval,F.f(input)];
end 
figure;
hold on
plot(iterations,Fval,'b')
xlabel('Number of iterations')
ylabel('Value of the objective function')
hold off


%% Convergence plots 
% repeat for PD and barrier

conPD = convergenceHistory(info, [], F, 2); % using 2-norm.
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
hold on, semilogy(find(info.xs > 1-eps), con.x(info.xs > 1-eps), 'xk', 'MarkerSize', 10);
legend('p=1', 'Full step')
end
% Linear convergence after initial phase for feasible x0
  
% This is not a right way of looking at suspected linear convergence.
% Algebraic convergence: ||x_{k} - x^*|| = c k^{-p}, c>0, bounded
% ln ||x_{k} - x^*|| = ln c + -p*(ln k)     [linear function in log log with slope -p and offset (ln c)]
%figure, loglog(con.x, 'r'); grid on; title('Algebraic convergence'); xlabel('ln(k)'); ylabel('ln(||x_{k} - x^*||')


