using Jellyfish

## reality check, make sure error is low when you feed in truth from SVD
M = lowrank(100, 100, 3)
U, S, V = svd(M)
M2 = jellyfish1(M, num_iter = 1000, L = U * sqrt(diagm(S)), R = (sqrt(diagm(S)) * V')')
@assert relerr(M, M2) < .01

