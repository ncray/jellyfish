function rel_err(X1, X2)
    normfro(X1 - X2) / normfro(X2)
end

function get_low_rank(n_r = 10, # number of rows
                      n_c = 10, # number of columns
                      r = 3)    # rank
    Y_l = randn(n_r, r)
    Y_r = randn(n_c, r)
    M = Y_l * Y_r'
    M / sqrt(mean(M.^2))
end

## basic version
## random sampling of rows/columns
function jellyfish_1(M,                        # target matrix
                     r = 3;                    # guess of rank
                     mu = 1,                   # regularization parameter
                     alpha_k = .01,            # learning rate
                     num_iter = 10000,         # number of iterations
                     L = randn(size(M, 1), r), # initial guess for L
                     R = randn(size(M, 2), r), # initial guess for R
                     all_errs = true) 
    n_r = size(M, 1)
    n_c = size(M, 2)
    rel_errs = zeros(num_iter)
    
    for k = 1:num_iter
        i = rand(1:n_r)
        j = rand(1:n_c)
        resid = alpha_k * 2 * (L[i, :] * R[j, :]' - M[i, j])
        L[i, :], R[j, :] = (1 - mu * alpha_k / n_r) * L[i, :] - resid * R[j, :],
        (1 - mu * alpha_k / n_c) * R[j, :] - resid * L[i, :]
        
        if all_errs
            rel_errs[k] = rel_err(L * R', M)
        end
    end
    #L * R', rel_errs ## REPL fills up when returning a tuple
    if all_errs
        rel_errs
    else
        rel_err(L * R', M)
    end
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

function gradient_updates(M, L, R, n_r, n_c, r, mu, alpha_k)
    n_r = size(M, 1)
    n_c = size(M, 2)

    for i in 1:n_r
        for j in 1:n_c
            resid = alpha_k * 2 * (L[i, :] * R[j, :]' - M[i, j])
            L[i, :], R[j, :] = (1 - mu * alpha_k / n_r) * L[i, :] - resid * R[j, :],
            (1 - mu * alpha_k / n_c) * R[j, :] - resid * L[i, :]
        end
    end
    (L, R)
end

#gradient_updates(X, 3, 1, .01)
#chunks = get_chunks(10, 10, 3)


## basic version
## permute rows/columns
function jellyfish_2(M,                        # target matrix
                     r = 3;                    # guess of rank
                     p = 3,                    # number of processors
                     mu = 1,                   # regularization parameter
                     alpha_k = .01,            # learning rate
                     num_iter = 10000,         # number of iterations
                     L = randn(size(M, 1), r), # initial guess for L
                     R = randn(size(M, 2), r), # initial guess for R
                     all_errs = true)
    n_r = size(M, 1)
    n_c = size(M, 2)
    chunks = get_chunks(n_r, n_c, p)

    for k = 1:num_iter
        for l in 0:(p-1)
            ##a = [1:p]
            ##b = [((x + l) % p) + 1 for x in a]
            ab = [(x, ((x + l) % p) + 1) for x in [1:p]]
            for (i, j) in ab
                inds = chunks[1][i], chunks[2][j]
                L[inds[1], 1:r], R[inds[2], 1:r] = gradient_updates(M[inds[1], inds[2]], L[inds[1], 1:r], R[inds[2], 1:r], n_r, n_c, r, mu, alpha_k)
            end
        end
    end
    L * R'
end














