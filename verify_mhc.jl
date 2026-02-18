
using Lux
using Random
using LinearAlgebra
using Test

include("src/model.jl")
using .Model

function verify_mhc_components()
    rng = Random.default_rng()
    Random.seed!(rng, 42)
    
    latent_dim = 64
    instr_dim = 32
    K = 4
    B = 2
    rank = 4

    @testset "Sinkhorn Knopp" begin
        println("Testing Sinkhorn-Knopp...")
        logits = randn(rng, Float32, K, K)
        M = Model.sinkhorn_knopp(logits)
        
        # Check doubly stochastic
        row_sums = sum(M, dims=2)
        col_sums = sum(M, dims=1)
        
        @test all(isapprox.(row_sums, 1.0; atol=1e-5))
        @test all(isapprox.(col_sums, 1.0; atol=1e-5))
        @test all(M .>= 0)
        println("Sinkhorn-Knopp PASSED")
    end

    @testset "InstructionGuidedReparameterization" begin
        println("Testing InstructionGuidedReparameterization...")
        layer = InstructionGuidedReparameterization(instr_dim, latent_dim, rank, K)
        ps, st = Lux.setup(rng, layer)
        
        mu = randn(rng, Float32, latent_dim, B)
        sigma = rand(rng, Float32, latent_dim, B)
        instr = randn(rng, Float32, instr_dim, B)
        
        # Forward
        z, st_new = layer((mu, sigma, instr), ps, st)
        
        # Check output shape: [Dim, K, B]
        @test size(z) == (latent_dim, K, B)
        println("InstructionGuidedReparameterization Output Shape: $(size(z)) -> PASSED")
    end

    @testset "MHCLatentReasoner" begin
        println("Testing MHCLatentReasoner...")
        reasoner = MHCLatentReasoner(latent_dim, K, 2, 4) # 2 layers, 4 heads
        ps, st = Lux.setup(rng, reasoner)
        
        # Input: [Dim, K, B]
        z_in = randn(rng, Float32, latent_dim, K, B)
        
        # Forward
        z_out, st_new = reasoner(z_in, ps, st)
        
        # Check output shape
        @test size(z_out) == (latent_dim, K, B)
        println("MHCLatentReasoner Output Shape: $(size(z_out)) -> PASSED")
        
        # Check mixing matrix initialization (should be close to uniform/identity if constraints hold, 
        # but pure random init might not strictly hold until Sinkhorn applied inside)
        # We verify that it runs without error.
    end
end

verify_mhc_components()
