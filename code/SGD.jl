# subgradient of obj
function stochastic_grad_obj(X, Y, W, A, C, lambda, i,j)
    N, D = size(X)
    K = size(W, 2)
    W = sparse(W)
    # i = rand(1:N)    # i -> examples
    # j = rand(1:K)    # j -> classes
    # @printf("i=%d, j=%d\n",i,j)
    Yij = Y[i,j]*2-1
    Xi = X[i,:]
    Wj = W[:,j]
    Aj = A[:,j]
    # L2-hinge loss
    gj = full(-2*C * max(0,1-Yij*Xi'*Wj) * Yij*Xi) * N*K
    # L1-regularization
    gj += lambda * sign.(Wj) * K
    # recursive regularization
    neighbours = find(Aj); l = length(neighbours)
    gn = zeros(D,l)
    for n = 1:l
        gn[:,n] = (W[:,neighbours[n]] - Wj) * K
        # @printf("    neighbour=%d\n",neighbours[n])
    end
    gj -= sum(gn,2)[:]
    # construct gradient
    rows = repmat(1:D, l+1)
    cols = zeros(Int64,(l+1)*D)
    for n = 1:l
        cols[(n-1)*D+1:n*D] = neighbours[n]
    end
    cols[l*D+1:(l+1)*D] = j
    G = sparse(rows,cols,[gn[:]; gj],D,K)
    return G, [neighbours;j]
end


function mainStochastic(X, Y, K, A, C, lambda;
                        stepsize=(i)->1e-8, maxIter = 1e3)
    N, D = size(X)
    K = size(Y, 2)
    #W = sparse([1],[1],[0.0],D,K)
    W = zeros(D, K)
    obj_val = obj(X, Y, W, A, C, lambda)
    println(@sprintf("Iter = %5d, Obj = %.5e ",0,obj_val))
    #objs = zeros(maxIter/500+1); objs[1] = obj_val
    #times = zeros(maxIter/500+1)
    objs = []
    times = []
    nzs = []
    tic()
    tt = 0
    for iter=1:maxIter
        i = rand(1:N)    # i -> examples
        j = rand(1:K)    # j -> classes
        G, active_cols = stochastic_grad_obj(X, Y, W, A, C, lambda, i, j)
        #W = W - stepsize(iter)*G
        for k in active_cols
            W[:,k] -= stepsize(iter)*G[:,k] 
            for jj = 1:D
                if W[jj,k] > stepsize(iter)*lambda
                    W[jj,k] -=  stepsize(iter)*lambda
                elseif  W[jj,k] < -stepsize(iter)*lambda  
                    W[jj,k] +=  stepsize(iter)*lambda
                else
                    W[jj,k] = 0
                end
            end
        end
        if (iter%500)==0
            #iter = Int(iter)
            #curr = Int(iter/500+1)
            #times[curr] = times[curr-1] + toc()
            obj_val = obj(X, Y, W, A, C, lambda)
            #objs[curr] = obj_val
            push!(objs, obj_val)
            tt += toq()
            push!(times, tt)
            nz = 0
            for kk = 1:K
                for jj = 1:D
                    if W[jj,kk] != 0 
                       nz += 1 
                    end
                end
            end
            tic()
            push!(nzs, nz)
            println(@sprintf("time = %5d, Iter = %5d, i = %d, j = %d, Obj = %.5e ",tt,iter,i,j,obj_val))
            #tic()
        end
    end
    toc();
    return W, times, objs, nzs
end
