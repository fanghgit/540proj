include("misc.jl")
# Hessian-vector multiplication


function Ha(X, Y, W, L, XW, a, C, lambda)
    N, D = size(X)
    K = size(Y, 2)
    XT = X'
    res = zeros(D*K)
    Ma = reshape(a, D, K)
    
    XMa = X*Ma
    
    #rows = rowvals(L)
    #vals = nonzeros(L)
    for k = 1:K
        #tmp = nzrange(L, k)
        Lk = L[:,k]
        neighbours, vv = findnz(Lk)
        #neighbours = rows[tmp]
        #vv = vals[tmp]
        for pp = 1:length(neighbours)
            res[ ((k-1)*D+1):(k*D)] += 2*a[ ((neighbours[pp]-1)*D+1):(neighbours[pp]*D)]*vv[pp]
        end
        #res[(k-1)*D +  ]
    end
    
    for k = 1:K
        Xa = XMa[:,k]
        score_k = XW[:,k]
        Yk = full(Y[:,k])
        Yk = (Yk-0.5)*2
        tmp = 1 - Yk.*score_k
        for i = 1:N
            if tmp[i] > 0
                xi = XT[:,i]
                res[((k-1)*D+1):(k*D)] += 2*C*Xa[i]*xi 
            end
        end
        
    end
    
    return res
end
    
    
function proximal_newton(X, Y, A, C, lambda, tol=1e-4, eps = 1e-4, maxiter = 100)
    N, D = size(X)
    K = size(Y, 2)
    XT = X'
    Y_full = 2*(full(Y)-0.5)
    
    W = zeros(D, K)
    
    # get graph laplacian
    Diag = zeros(K)
    for i = 1:K
        Diag[i] = sum( A[:,i] ) 
    end
    DG = diagm(Diag)
    #@show(size(A))
    #@show(size(DG))
    L = sparse(DG - A)
    
    total_time = 0
    
    #output
    time_list = []
    obj_list = []
    nz_list = []
    
    tic()
    iter = 1
    for iter = 1:maxiter
        
        
        
        #if iter >= 2
        #    XW_2 = X*W
        #    vv = vecnorm(XW_2 - XW, 2)^2
        #    @printf "vv: %f\n" vv
        #end
        # construct diagnoal H
        diagH = zeros(round(D*K))
        for k = 1:K
            #@show(Diag[k])
            diagH[ round((k-1)*D+1): round(k*D)] = 2*Diag[k]
        end
        
        XW = X*W
        XDelta = zeros(N,K)
        
        for k = 1:K
            wk = W[:,k]
            Yk = full(Y[:,k])
            Yk = (Yk-0.5)*2
            #score_k = X*wk
            #XW[:,k] = score_k
            score_k = XW[:,k]
            tmp = score_k.*Yk
            #@show(size(tmp))
            for i = 1:N
                if 1 - tmp[i] > 0
                    xi = XT[:,i]
                    idx, vals = findnz(xi)
                    for pp = 1:length(idx)
                        diagH[ (k-1)*D + idx[pp] ] += 2*C*vals[pp]^2 
                    end
                end
            end
        end
        
        g = get_grad( X, Y, W, L, C, lambda )
        
        #get active set
        rows = zeros(Int64, 0)
        cols = zeros(Int64, 0)
        nz = 0
        for j = 1:D
            for k = 1:K
                if abs(g[j,k]) > lambda - eps || W[j,k] != 0
                    push!(rows, j)
                    push!(cols, k)
                end
                if W[j,k] != 0
                   nz += 1 
                end
            end
        end
        @printf "size of active set: %d\n" length(rows)
        #coordinate update
        #w = vec(W)
        obj_old = obj(X, Y, W, A, C, lambda)
        
        
        total_time += toq()
        tic()
        
        push!(time_list, total_time)
        push!(obj_list, obj_old)
        push!(nz_list, nz)
        
        
        #permutation
        perm = randperm(length(rows))
        rows = rows[perm]
        cols = cols[perm]
        
        ### check 
        
        check_approx1 = lambda*vecnorm(W, 1) 
        #check_d = zeros(D,K)
        Delta = zeros(D,K)
        
        #H_check = 2*kron(L, eye(D))
        #for k = 1:K
        #    for i = 1:N
        #        xi = XT[:,i]
        #        if 1 - Y_full[i,k]*XW[i,k] > 0
        #            H_check[ ((k-1)*D+1):(k*D), ((k-1)*D+1):(k*D) ] += 2*C*xi*xi'
        #        end
        #    end
        #     
        #end
        
        ###
        #tic()
        @printf "start coordinate descent!\n"
        
        #for inner_iter = 1:3
        
        for l = 1:length(rows)
            j = rows[l]
            k = cols[l]
            
            idx = round( (k-1)*D+j )
            h0 = diagH[ idx ]
    
            deno = g[j,k]
    
            # consider the k-th block
            H0w = 0
            Lk = L[:,k]
            idxs, vals = findnz(Lk)
            
            #check
            h0_check = 0
            for pp = 1:length(idxs)
                kk = idxs[pp]
                H0w += 2*Delta[j,kk]*vals[pp] 
                #check
                #if kk == k
                #    h0_check += 2*vals[pp]
                #end
            end
    
            xj = X[:,j]
            idxs, vals = findnz(xj)
            for pp = 1:length(idxs)
                ii = idxs[pp]
                    # time consuming?
                if 1 - Y_full[ii,k]*XW[ii,k] > 0
                    H0w += 2*C*vals[pp]*XDelta[ii,k] 
                    #check 
                    #h0_check += 2*C*vals[pp]^2
                end
            end
            
            #if abs(h0_check - h0) > 1e-5
            #    @printf "h0: %f, h0_check: %f, H_check[]: %f\n" h0 h0_check H_check[idx,idx]
            #    error("h0 check fails") 
            #end
            
            #### check H0w
            #H0w_check = dot( H_check[ (k-1)*D+j ,:], vec(Delta) )
            #if abs(H0w - H0w_check) > 1e-4
            #    @printf "H0w_check: %f, H0w: %f\n" H0w_check H0w
            #    error("H0w check fails") 
            #end
            
            #Hd = Ha(X, Y, W, L, XW, vec(W), C, lambda)
            #Hd_check = H_check*vec(W)
            #if vecnorm(Hd_check - Hd) > 1e-4
            #    error("Ha check fails") 
            #end
            
            
    
            H0w -= h0* Delta[j,k]
            
            ### check
            #H0w = (H_check*vec(check_d))[idx]
            #H0w -= H_check[idx, idx]*check_d[j,k]
            #h0 = H_check[idx, idx]
            
            deno += H0w
            if h0 <= 0
                @printf "h0: %f\n" h0
                error("h0 < 0 ") 
            end
            delta_jk = -deno/h0
            w_jk = W[j,k] + delta_jk
            if w_jk > lambda/h0
                w_jk -= lambda/h0 
            elseif w_jk < -lambda/h0
                w_jk += lambda/h0
            else
                w_jk = 0
            end
            
            diff = w_jk - W[j,k]
            
            obj1 = lambda*abs(W[j,k])
            obj2 = h0/2*diff^2 + diff*deno + lambda*abs(W[j,k]+diff)
            
            if obj1 < obj2 - 1e-4
                @printf "check coordinate descent: %f, %f" obj1 obj2
                @printf "l: %d\n" l
                #error("coordinate descent error") 
            end
            
            #@printf "check coordinate descent: %f, %f" obj1 obj2
            
            
            W[j,k] = w_jk
            
            
            #update XDelta
            #XDelta[:,k] += (diff*xj)
            for pp = 1:length(idxs)
                ii = idxs[pp]
                XDelta[ii,k] += diff*vals[pp]
            end
            
            
            Delta[j,k] += diff
            
        end
            
        #end
        #tt_cd = toq()
        #@printf "CD time: %f \n" tt_cd
        
        ### check approx
        
        #check_approx2 = lambda*vecnorm(W, 1) + dot(vec(g), vec(check_d))
        #Hd = Ha(X, Y, W, L, XW, vec(check_d), C, lambda)
        #check_approx2 += 0.5*( dot(vec(check_d), Hd) )
        #@printf "dot: %f\n" dot(vec(g), vec(check_d))
        #@printf "hessian part: %f\n" 0.5*( dot(vec(check_d), Hd) )
        #@printf "approx1: %f, approx2: %f\n" check_approx1 check_approx2
        
        ### dense checker
        #check_approx2 = 0.5* dot( vec(Delta), H_check*vec(Delta) ) + dot(vec(g), vec(Delta) ) + lambda*vecnorm(W,1)
        #@printf "approx1: %f, approx2: %f\n" check_approx1 check_approx2
        
        
        
        obj_new = obj(X, Y, W, A, C, lambda)
        @printf "time: %f, obj_old: %f, obj_new: %f \n" total_time obj_old obj_new
         
    end
    
    #writedlm("rcv1_newton.csv", [time_list obj_list nz_list], ',');

    
    return W, time_list, obj_list, nz_list;
end

function proximal_gradient(X, Y, A, C, lambda, tol=1e-4, maxiter = 100, max_inner_iter = 20)
    N, D = size(X)
    K = size(Y, 2)
    
    W = zeros(D, K)
    # get graph laplacian
    Diag = zeros(K)
    for i = 1:K
        Diag[i] = sum( A[:,i] ) 
    end
    DG = diagm(Diag)
    L = sparse( DG - A )
    
    time_list = []
    obj_list = []
    nz_list = []
    
    tic()
    
    tt = 0
    # iterative gradient
    iter = 1
    for iter = 1:maxiter
        g = get_grad( X, Y, W, L, C, lambda ) 
        @printf "get gradient complete\n"
        # prox step
        #d = zeros(D)
        alpha = 1
        obj_old = obj(X, Y, W, A, C, lambda)
        
        tt += toq()
        push!(time_list, tt)
        push!(obj_list, obj_old)
        nz = 0
        for k=1:K
            for j=1:D
                if W[j,k] != 0
                   nz += 1 
                end
            end
        end
        push!(nz_list, nz)
        
        @printf "time: %f, obj: %f, nz: %f\n" tt obj_old nz
        
        
        tic()
        
        @printf "obj complete\n"
        for inner_iter = 1:max_inner_iter
            threshold = alpha*lambda
            
            W_new = W - alpha*g 
            for jj = 1:D
                for k = 1:K
                    if W_new[jj, k] > threshold
                        W_new[jj, k] -= threshold
                    elseif W_new[jj, k] < -threshold
                        W_new[jj, k] += threshold
                    else
                        W_new[jj,k] = 0
                    end
                end
            end
            @printf "prox complete\n"
            obj_new = obj(X, Y, W_new, A, C, lambda)
            if obj_new < obj_old
                W = W_new
                println("stepsize: ", alpha)
                break 
            end
            alpha = alpha/10
            if inner_iter == max_inner_iter
                error("linesearch failed") 
            end
        end
        
        @printf "iter %d finished!\n" iter
        
    end
    return W, time_list, obj_list, nz_list
end