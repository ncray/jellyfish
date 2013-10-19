module Jellyfish

include("utils.jl")

export jellyfish_1, rel_err, get_low_rank

## basic version
## random sampling of rows/columns
function jellyfish_1(M,                        # target matrix
                     r = 3;                    # guess of rank
                     mu = 1,                   # regularization parameter
                     alpha_k = .01,            # learning rate
                     num_iter = 10000,         # number of iterations
                     L = randn(size(M, 1), r), # initial guess for L
                     R = randn(size(M, 2), r)) # initial guess for R
    n_r = size(M, 1)
    n_c = size(M, 2)
    rel_errs = zeros(num_iter)
    
    for k = 1:num_iter
        i = rand(1:n_r)
        j = rand(1:n_c)
        resid = alpha_k * 2 * (L[i, :] * R[j, :]' - M[i, j])
        L[i, :], R[j, :] = (1 - mu * alpha_k / n_r) * L[i, :] - resid * R[j, :],
        (1 - mu * alpha_k / n_c) * R[j, :] - resid * L[i, :]
    end
    
    (L * R', L, R)
end

## basic version
## new permutation on each iteration of rows/columns
function jellyfish_2(M,                        # target matrix
                     r = 3;                    # guess of rank
                     mu = 1,                   # regularization parameter
                     alpha_k = .01,            # learning rate
                     num_iter = 10,            # number of iterations
                     L = randn(size(M, 1), r), # initial guess for L
                     R = randn(size(M, 2), r)) # initial guess for R
    n_r = size(M, 1)
    n_c = size(M, 2)
    rel_errs = zeros(num_iter)
    
    for k = 1:num_iter
        perm_i = randperm(n_r)
        perm_j = randperm(n_c)
        for i in perm_i
            for j in perm_j
                resid = alpha_k * 2 * (L[i, :] * R[j, :]' - M[i, j])
                L[i, :], R[j, :] = (1 - mu * alpha_k / n_r) * L[i, :] - resid * R[j, :],
                (1 - mu * alpha_k / n_c) * R[j, :] - resid * L[i, :]
            end
        end
    end
    
    (L * R', L, R)
end

function get_chunks(n_r, n_c, p)
    perm_r = randperm(n_r)
    perm_c = randperm(n_c)

    C_a = Dict{Int, Array{Int64, 1}}()
    C_b = Dict{Int, Array{Int64, 1}}()
    for i in 1:p
        C_a[i] = Int[]
        C_b[i] = Int[]
    end

    for i in 1:n_r
        push!(C_a[int(floor(p / n_r * (perm_r[i] - 1)) + 1)], i)    
    end

    for j in 1:n_c
        push!(C_b[int(floor(p / n_c * (perm_c[j] - 1)) + 1)], j)
    end

    (C_a, C_b)
end

function gradient_updates(M, L, R, r, mu, alpha_k, n_r, n_c)
    n_r_L = size(L, 1)
    n_c_R = size(R, 1)
    
    for i in 1:n_r_L
        for j in 1:n_c_R
            resid = alpha_k * 2 * (L[i, :] * R[j, :]' - M[i, j])
            L[i, :], R[j, :] = (1 - mu * alpha_k / n_r) * L[i, :] - resid * R[j, :],
            (1 - mu * alpha_k / n_c) * R[j, :] - resid * L[i, :]
        end
    end
    
    (L, R)
end

## M = reshape(linspace(1, 100, 100), 10, 10)'
## function foo(x)
##     x[1] = -1
## end
## foo(M[1, :])
## M
#gradient_updates(X, 3, 1, .01)
#chunks = get_chunks(10, 10, 3)

function jellyfish_3(M,                        # target matrix
                     r = 3;                    # guess of rank
                     p = 3,                    # number of processors
                     mu = 1,                   # regularization parameter
                     alpha_k = .01,            # learning rate
                     num_iter = 10,            # number of iterations
                     L = randn(size(M, 1), r), # initial guess for L
                     R = randn(size(M, 2), r)) # initial guess for R
    n_r = size(M, 1)
    n_c = size(M, 2)
    chunks = get_chunks(n_r, n_c, p)
    
    for k = 1:num_iter
        for l in 0:(p - 1)
            ##a = [1:p]
            ##b = [((x + l) % p) + 1 for x in a]
            ab = [(x, ((x + l) % p) + 1) for x in [1:p]]
            for (i, j) in ab
                inds = chunks[1][i], chunks[2][j]
                L[inds[1], 1:r], R[inds[2], 1:r] = gradient_updates(M[inds[1], inds[2]], L[inds[1], 1:r], R[inds[2], 1:r], r, mu, alpha_k, n_r, n_c)
            end
        end
    end
    (L * R', L, R)
end

## X = {rand(1000,1000), rand(1000,1000), rand(1000,1000), rand(1000,1000)}
## @elapsed pmap(svd, X)
## @elapsed svd(rand(1000, 1000))

## addprocs(3)
## tic()
## @parallel for i = 1:4
##     svd(rand(2000, 2000))
## end
## s = toc()
## println(s)

## @elapsed svd(rand(2000, 2000))

## pmap((x)->x^2, [1:4])

function jellyfish_4(M,                        # target matrix
                     r = 3;                    # guess of rank
                     p = 3,                    # number of processors
                     mu = 1,                   # regularization parameter
                     alpha_k = .01,            # learning rate
                     num_iter = 10,            # number of iterations
                     L = randn(size(M, 1), r), # initial guess for L
                     R = randn(size(M, 2), r)) # initial guess for R
    n_r = size(M, 1)
    n_c = size(M, 2)
    chunks = get_chunks(n_r, n_c, p)
    
    for k = 1:num_iter
        for l in 0:(p - 1)
            ab = [(x, ((x + l) % p) + 1) for x in [1:p]]
            refs = RemoteRef[]
            
            for (i, j) in ab
                inds = chunks[1][i], chunks[2][j]
                push!(refs, @spawn gradient_updates(M[inds[1], inds[2]], L[inds[1], 1:r], R[inds[2], 1:r], r, mu, alpha_k, n_r, n_c))
            end
            
            for (i, j) in ab
                inds = chunks[1][i], chunks[2][j]
                L[inds[1], 1:r], R[inds[2], 1:r] = fetch(shift!(refs))
            end                
        end
    end
    (L * R', L, R)
end

## function grad_wrapper(inds)
##     gradient_updates(M[inds[1], inds[2]], L[inds[1], 1:r], R[inds[2], 1:r], r, mu, alpha_k, n_r, n_c)
## end

## function jellyfish_4(M,                        # target matrix
##                      r = 3;                    # guess of rank
##                      p = 3,                    # number of processors
##                      mu = 1,                   # regularization parameter
##                      alpha_k = .01,            # learning rate
##                      num_iter = 10,            # number of iterations
##                      L = randn(size(M, 1), r), # initial guess for L
##                      R = randn(size(M, 2), r)) # initial guess for R
##     n_r = size(M, 1)
##     n_c = size(M, 2)
##     chunks = get_chunks(n_r, n_c, p)

##     for k = 1:num_iter
##         for l in 0:(p - 1)
##             ab = [(x, ((x + l) % p) + 1) for x in [1:p]]
##             all_inds = [(chunks[1][i], chunks[2][j]) for (i, j) in ab]
##                 res = pmap(grad_wrapper, all_inds)
##                 counter = 1
##             for (i, j) in ab
##                 inds = chunks[1][i], chunks[2][j]
##                 L[inds[1], 1:r], R[inds[2], 1:r] = res[counter]
##                 counter += 1
##             end
##         end
##     end
##     (L * R', L, R)
## end


end

