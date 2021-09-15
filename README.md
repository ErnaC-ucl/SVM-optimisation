# SVM-optimisation
Optimisation of the SVM algorithm using Primal-Dual Interior Points Method and SMO

Support vector machines (SVMs) are a well established and rigorously funded technique for the solution
of classification problems in machine learning. In this report the SVM optimisation problem for binary classification is solved using two different methods:
- 1. Primal-Dual Interior Points Method;
- 2. Sequential Minimal Optimisation Method.

We observe that the Primal-Dual method displays a better performance than the SMO for our
SVM optimisation problem. The performance is evaluated in terms of computational efficiency
(i.e. how expensive each method is in terms of computational time and numbers of iterations) and
classification accuracy. The relative over-performance of the primal dual method is attributed to the following factors:
- Primal-dual methods are proved to be reliable and accurate optimisation techniques for small
and moderately sized problems. Whereas, the SMO method is more applicable for large-scale
problems as the main advantage of the SMO lies in the fact that solving for two Lagrange
multipliers can be done analytically, numerical optimization is avoided entirely.
- Our optimisation problem is solved efficiently by the primal dual method because this method
asymptotically achieves super-linear convergence. Whereas, for a general kernel matrix D,
the SMO method achieves only sublinear convergence.
- Finally,Primal-Dual methods provide a guarantee to solve optimization problems
in O(sqrt(n)log(1/ε)) iterations and, in practice, display a fast convergence in merely a few
iterations almost independent of the problem size. Global convergence is also guaranteed for
the SMO method with a general kernel matrix, however in our implementation we are using
a simplified version of the SMO algorithm for which the global convergence is no longer
guaranteed.

**In this repository you can find:**
- the pdf report summarising the methodology and results;
- Matlab code for the implementation of the two methods;
