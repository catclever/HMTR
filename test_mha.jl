using Lux, Random, NNlib, LinearAlgebra
dim = 16
heads = 2
L = 5
B = 3
mha = MultiHeadAttention(dim; nheads=heads)
rng = Random.default_rng()
ps, st = Lux.setup(rng, mha)
x = randn(Float32, dim, L, B)
mask = tril(ones(Bool, L, L))
try
    y, st_out = mha((x, x, x, mask), ps, st)
    println("SUCCESS with 4-tuple mask")
catch e
    println("ERROR: ", e)
end
