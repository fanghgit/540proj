# Solvers for Hierarchical Classification
## Description
This repo currently includes the following folders:
- data: folder to hold experimental data, currently only have diatoms and reuters, both are taken from [1].
- util: contains code for preprocessing and evalution.
- code: proximal-newton, proximal-gradient and proximal-sgd solver for hierarchical classification
- experimental_result: hold some output and figures.

## Quick Start
Run the following code to load data:

```
julia > include("code/readData.jl")
```
Load data matrix X, label matrix Y and labels' adjacency matrix A.
```
julia > X, Y = read("data/rcv1/train_remap.txt");
julia > K = size(Y, 1)
julia > A = read_cat_hier("data/rcv1/hr_remap.txt",K)
julia > Y = Y'
```
Add bias term to our data matrix X.
```
julia > N, D = size(X)
julia > X = [ones(N) X]
julia > K = size(Y,2)
```
Set hyper-parameters
```
julia > C = 1; lambda = 0.1
```
Run proximal-newton:
```
julia > include("code/prox_newton.jl")
julia > W = proximal_newton(X, Y, A, C, lambda,1e-4, 1e-4,200)
```
To run proximal-gradient:
```
julia > include("code/prox_newton.jl")
julia > W = proximal_gradient(X, Y, A, C, lambda, 1e-4, 300, 20)
```
To run proximal-SGD
```
julia > include("code/SGD.jl")
julia > W_res = mainStochastic(X, Y, K, A, C, lambda,stepsize=(i)->1e-8, maxIter=2e4)
```
References:
[1] HR-SVM Project, https://sites.google.com/site/hrsvmproject/datasets-hier
