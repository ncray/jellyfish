using Jellyfish

M = get_low_rank(100, 100, 3)
U, S, V = svd(M)

## reality check, make sure error is low when you feed in truth from SVD
M2 = jellyfish_1(M, num_iter = 1000, L = U * sqrt(diagm(S)), R = (sqrt(diagm(S)) * V')')[1]
@assert rel_err(M, M2) < .01
