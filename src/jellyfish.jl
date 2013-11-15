module Jellyfish

using NumericExtensions
include("utils.jl")

export rel_err, get_low_rank, get_chunks, jellyfish_1, jellyfish_2, jellyfish_3, jellyfish_4

## basic version
## random sampling of rows/columns
function jellyfish1(X,                        # target matrix
                    r = 3;                    # guess of rank
                    mu = 1,                   # regularization parameter
                    alpha_k = .01,            # learning rate
                    num_iter = 10000,         # number of iterations
                    L = randn(size(X, 1), r), # initial guess for L
                    R = randn(size(X, 2), r)) # initial guess for R
    I, J, K = findnz(X)
    nr = size(X, 1)
    nc = size(X, 2)
    weightrow = zeros(nr)
    weightcol = zeros(nc)
    for i in 1:length(I)
        weightrow[I[i]] += 1
        weightcol[J[i]] += 1
    end
    
    for l in rand(1:length(I), num_iter)
        i = I[l]
        j = J[l]
        resid = alpha_k * 2 * (L[i, :] * R[j, :]' - X[i, j])
        L[i, :], R[j, :] = (1 - mu * alpha_k / weightrow[i]) * L[i, :] - resid * R[j, :],
        (1 - mu * alpha_k / weightcol[j]) * R[j, :] - resid * L[i, :]
    end
    
    L * R'
end

function get_chunks(nr, nc, p)
    perm_r = randperm(nr)
    perm_c = randperm(nc)

    C_a = Dict{Int, Array{Int64, 1}}()
    C_b = Dict{Int, Array{Int64, 1}}()
    for i in 1:p
        C_a[i] = Int[]
        C_b[i] = Int[]
    end

    for i in 1:nr
        push!(C_a[int(floor(p / nr * (perm_r[i] - 1)) + 1)], i)    
    end

    for j in 1:nc
        push!(C_b[int(floor(p / nc * (perm_c[j] - 1)) + 1)], j)
    end

    (C_a, C_b)
end

## function gradient_updates(X, L, R, r, mu, alpha_k, nr, nc)
##     nr_L = size(L, 1)
##     nc_R = size(R, 1)
    
##     for i in 1:nr_L
##         for j in 1:nc_R
##             if X[i, j] != 0
##                 resid = alpha_k * 2 * (L[i, :] * R[j, :]' - X[i, j])
##                 L[i, :], R[j, :] = (1 - mu * alpha_k / nr) * L[i, :] - resid * R[j, :],
##                 (1 - mu * alpha_k / nc) * R[j, :] - resid * L[i, :]
##             end
##         end
##     end
    
##     (L, R)
## end

function gradient_updates(X::Array{Float64,2}, L::Array{Float64,2}, R::Array{Float64,2}, r::Int64, mu::Float64, alpha_k::Float64, n_r::Int64, n_c::Int64)
    n_r_L = size(L, 1)
    n_c_R = size(R, 1)
    alpha_k_times_two = alpha_k * 2
    l_shrinkage = (1 - mu * alpha_k / n_r)
    r_shrinkage = (1 - mu * alpha_k / n_c)
    l = zeros(Float64, r)
    rr = zeros(Float64, r)
    
    for i in 1:n_r_L
        for j in 1:n_c_R
            if X[i, j] != 0
                tmp = 0.0
                for k in 1:r
                    @inbounds tmp += L[i, k] * R[j, k]
                end
                @inbounds resid = alpha_k_times_two * (tmp - X[i, j])
                for k in 1:r
                    @inbounds l[k] = l_shrinkage * L[i, k] - resid * R[j, k]
                end
                for k in 1:r
                    @inbounds R[j, k] = r_shrinkage * R[j, k] - resid * L[i, k]
                end
                @inbounds L[i, :] = l
            end
        end
    end
    
    (L, R)
end

function jellyfish2(X,                        # target matrix
                    r = 3;                    # guess of rank
                    p = 3,                    # number of processors
                    mu = 1.,                  # regularization parameter
                    alpha_k = .01,            # learning rate
                    num_iter = 10,            # number of iterations
                    L = randn(size(X, 1), r), # initial guess for L
                    R = randn(size(X, 2), r)) # initial guess for R
    nr = size(X, 1)
    nc = size(X, 2)
    chunks = get_chunks(nr, nc, p)
    
    for k = 1:num_iter
        for l in 0:(p - 1)
            ab = [(x, ((x + l) % p) + 1) for x in [1:p]]
            refs = RemoteRef[]
            
            for (i, j) in ab
                inds = chunks[1][i], chunks[2][j]
                push!(refs, @spawn gradient_updates(X[inds[1], inds[2]], L[inds[1], 1:r], R[inds[2], 1:r], r, mu, alpha_k, nr, nc))
            end
            
            for (i, j) in ab
                inds = chunks[1][i], chunks[2][j]
                L[inds[1], 1:r], R[inds[2], 1:r] = fetch(shift!(refs))
            end                
        end
    end
    L * R'
end


end

