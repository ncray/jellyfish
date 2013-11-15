using Jellyfish

x = lowrank(500, 500, 3)
xs = jsample(x)
@time relerr(x, svdimpute(xs, 3, 100))
@time relerr(x, jellyfish1(xs, num_iter = 100000, alpha_k = .09))
@time relerr(x, jellyfish2(xs, num_iter = 20, alpha_k = .05, p = 3))
