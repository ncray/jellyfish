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
        
        rel_errs[k] = rel_err(L * R', M)
    end
    #L * R', rel_errs ## REPL fills up when returning a tuple
    rel_errs
end

