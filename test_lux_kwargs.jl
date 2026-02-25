using Lux, Random
sc = Chain(Dense(2,2), Dense(2,2))
rng = Random.default_rng()
ps, st = Lux.setup(rng, sc)
x = randn(Float32, 2,2)
try
    sc(x, ps, st; mask=1)
    println("SUCCESS")
catch e
    println("ERROR: ", typeof(e))
end
