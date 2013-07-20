Pkg.update()
Pkg.add("Benchmark")
Pkg.add("GZip")
Pkg.add("Gadfly")
Pkg.add("Color")
Pkg.add("DataFrames")

Pkg.add("Options")
Pkg.add("Calculus")
Pkg.add("Distributions")
Pkg.add("Stats")



include("jellyfish.jl")

M = get_low_rank(100, 100, 3)
U, S, V = svd(M)

## reality check, make sure error is low when you feed in truth from SVD
rel_errs = jellyfish_1(M, num_iter = 100, L = U * sqrt(diagm(S)), R = (sqrt(diagm(S)) * V')')

rel_errs = jellyfish_1(M, num_iter = 100000)
## rel_err(Mhat, M)
## rel_err(randn(size(M, 1), size(M, 2)), M)

using Gadfly
using DataFrames

df = DataFrame(x = 1:length(rel_errs), y = rel_errs)
p = plot(df, x = "x", y = "y", Geom.line)
draw(D3("jellyfish.js", 6inch, 6inch), p)


using Benchmark
M = get_low_rank(100, 100, 3)
j1() = jellyfish_1(M, num_iter = 100000, all_errs = false)
benchmark(j1, "j1", 3)


@time jellyfish_1(M, num_iter = 100000, all_errs = false)
@time M2 = jellyfish_2(M, num_iter = 8)
rel_err(M, M2)
