function rel_err(X1, X2)
    normfro(X1 - X2) / normfro(X2)
end

function get_low_rank(n_r = 10, # number of rows
                      n_c = 10, # number of columns
                      r = 3)    # rank
    Y_l = randn(n_r, r)
    Y_r = randn(n_c, r)
    M = Y_l * Y_r'
    M / sqrt(mean(M .^ 2))
end
