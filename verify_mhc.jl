using Pkg
Pkg.activate(".")
using Lux
using Random
using Test
using Zygote
include("src/HMTR.jl")
using .HMTR.Model
import .HMTR.Model: InitialMixingLayer, MHCLatentReasoner, MHCBlock

function test_mhc_forward()
    rng = Random.default_rng()
    Random.seed!(rng, 42)
    
    dim = 64
    heads = 4
    K = 4
    L = 16
    B = 2
    
    println("--- Testing InitialMixingLayer ---")
    mixer = InitialMixingLayer(dim, K)
    ps_mix, st_mix = Lux.setup(rng, mixer)
    
    # Input is from Stage 1: [Dim, L, B]
    x_in = randn(rng, Float32, dim, L, B)
    x_k, st_mix = mixer(x_in, ps_mix, st_mix)
    
    @test size(x_k) == (dim, K, L, B)
    println("InitialMixingLayer Output Shape: ", size(x_k), " ✓")
    
    println("\n--- Testing MHCLatentReasoner ---")
    reasoner = MHCLatentReasoner(dim, K, heads, 2)
    ps_reas, st_reas = Lux.setup(rng, reasoner)
    
    # Forward pass
    y, st_reas = reasoner(x_k, ps_reas, st_reas)
    @test size(y) == (dim, K, L, B)
    println("MHCLatentReasoner Output Shape: ", size(y), " ✓")
    
    println("\n--- Testing Sinkhorn-Knopp Properties ---")
    # Verify the learned M matrix in the first block is doubly stochastic
    M_raw = ps_reas.blocks.layer_1.M_raw
    M_norm = Model.sinkhorn_knopp(M_raw)
    
    row_sums = sum(M_norm; dims=2)
    col_sums = sum(M_norm; dims=1)
    
    # Tolerances are somewhat loose because of float32 and few iterations
    @test all(isapprox.(row_sums, 1.0f0; atol=1e-3))
    @test all(isapprox.(col_sums, 1.0f0; atol=1e-3))
    println("Row Sums: ", vec(row_sums))
    println("Col Sums: ", vec(col_sums))
    println("Doubly Stochastic Constraints Satisfied ✓")
    
    println("\n--- Testing Gradient Flow (Zygote) ---")
    # Just a simple scalar loss to ensure no mutating array errors in backward pass
    loss, back = Zygote.pullback(ps_reas) do p
        out, _ = reasoner(x_k, p, st_reas)
        sum(abs2, out)
    end
    
    grads = back(1.0f0)[1]
    # Check if M_raw got gradients
    @test grads.blocks.layer_1.M_raw !== nothing
    println("Backward pass successful. M_raw gradient exists ✓")
end

test_mhc_forward()
