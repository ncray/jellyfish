# Returns the relative Frobenius error between two matrices.
function relerr(X1, X2)
    normfro(X1 - X2) / normfro(X1)
end

# Generates a low-rank matrix normalized so that the average squared magnitude is 1.
function lowrank(nr = 10, # number of rows
                 nc = 10, # number of columns
                 r = 3)   # rank
    Xl = randn(nr, r)
    Xr = randn(nc, r)
    X = Xl * Xr'
    X / sqrt(mean(X .^ 2))
end

# Returns a dense, sampled matrix (zeroes indicate lack of sampling).
function jsample(X,           # input matrix
                 r = rank(X), # rank of input matrix
                 beta = 5)    # sampling multiple
    nr = size(X, 1)
    nc = size(X, 2)
    # number to sample
    n = beta * r * (nr + nc - r)

    # sample n entries uniformly
    pairs = shuffle(reshape([(i, j) for i in [1:nr], j in [1:nc]], 1, nr * nc)[:])[1:n]
    # zero out the rest
    ret = zeros(nr, nc)
    for (i, j) in pairs
       ret[i, j] = X[i, j]
    end
    
    ret
end

# Returns truncated SVD.
function truncsvd(X, # input matrix
                  r) # rank of truncated SVD
    U, S, V = svd(X)
    S[(r + 1):end] = 0
    U * diagm(S) * V'
end

# Returns SVD-imputed (on dense, zeroed-out, sampled matrix) matrix.
function svdimpute(X,          # input matrix
                   r,          # rank to use for imputation
                   niter = 10) # number of iterations
    I, J, K = findnz(X)
    ret = truncsvd(X, r)

    for i in 1:niter
        for i in 1:length(I)
            ret[I[i], J[i]] = K[i]
        end
        ret = truncsvd(ret, r)
    end

    ret
end

