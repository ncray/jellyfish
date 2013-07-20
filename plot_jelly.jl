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
M2 = jellyfish_1(M, num_iter = 100, L = U * sqrt(diagm(S)), R = (sqrt(diagm(S)) * V')')[1]
@assert rel_err(M, M2) < .01

using Benchmark
nrow = 100
ncol = 100
num_iter = 10
M = get_low_rank(nrow, ncol, 3)

rel_err(jellyfish_1(M, num_iter = num_iter * nrow * ncol)[1], M)
rel_err(jellyfish_2(M, num_iter = num_iter)[1], M)

j1() = jellyfish_1(M, num_iter = num_iter * nrow * ncol)
j2() = jellyfish_2(M, num_iter = num_iter)
benchmark(j1, "j1", 3)
benchmark(j2, "j2", 3)

compare([j1, j2], 4)

## using Gadfly
## using DataFrames

## df = DataFrame(x = 1:length(rel_errs), y = rel_errs)
## p = plot(df, x = "x", y = "y", Geom.line)
## draw(D3("jellyfish.js", 6inch, 6inch), p)

