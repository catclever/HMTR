module Model

using Lux
using Random
using NNlib
using Zygote
using CUDA
using LuxCUDA
using Statistics
import ChainRulesCore
using LinearAlgebra

export FeatureLayerNorm, SimplifiedMambaBlock, MambaCompressor, LatentPredictor, NanoDecoder, HMTR_Stage1_AutoEncoder, MHCBlock, MHCLatentReasoner, InitialMixingLayer

hippo_A_diag(d_state::Int, ::Type{T}=Float32) where {T<:AbstractFloat} = -(T.(collect(1:d_state)))

function reparameterize(μ, logvar; rng=Random.default_rng(), training=true)
    if !training
        return μ
    end
    σ = exp.(0.5f0 .* logvar)
    μ_parent = μ
    while μ_parent isa SubArray || μ_parent isa Base.ReshapedArray
        μ_parent = parent(μ_parent)
    end
    ε = if μ_parent isa CUDA.AbstractGPUArray
        Zygote.dropgrad(CUDA.randn(eltype(μ), size(μ)))
    else
        Zygote.dropgrad(randn(rng, eltype(μ), size(μ)))
    end
    return μ .+ σ .* ε
end

struct FeatureLayerNorm <: Lux.AbstractLuxLayer
    dim::Int
    epsilon::Float32
end

FeatureLayerNorm(dim::Int; epsilon::Float32=1f-5) = FeatureLayerNorm(dim, epsilon)

Lux.initialparameters(rng::AbstractRNG, l::FeatureLayerNorm) = (
    scale=ones(Float32, l.dim, 1),
    bias=zeros(Float32, l.dim, 1),
)
Lux.initialstates(rng::AbstractRNG, l::FeatureLayerNorm) = NamedTuple()

function (l::FeatureLayerNorm)(x::AbstractArray, ps, st)
    denom = Float32(max(l.dim, 1))
    μ = sum(x; dims=1) ./ denom
    x0 = x .- μ
    σ2 = sum(abs2, x0; dims=1) ./ denom
    invσ = inv.(sqrt.(σ2 .+ l.epsilon))
    y = x0 .* invσ
    y = y .* reshape(ps.scale, l.dim, 1, 1) .+ reshape(ps.bias, l.dim, 1, 1)
    return y, st
end

# --- Helper Layers ---

struct LatentPredictor{L<:Lux.AbstractLuxLayer} <: Lux.AbstractLuxLayer
    layer::L
end

function LatentPredictor(dim::Int)
    return LatentPredictor(Dense(dim => dim))
end

Lux.initialparameters(rng::AbstractRNG, l::LatentPredictor) = (
    layer=Lux.initialparameters(rng, l.layer),
)

Lux.initialstates(rng::AbstractRNG, l::LatentPredictor) = (
    layer=Lux.initialstates(rng, l.layer),
)

function (l::LatentPredictor)(x, ps, st)
    Dim, Lcap, B = size(x)
    x_flat = reshape(x, Dim, Lcap * B)
    y_flat, st_layer = l.layer(x_flat, ps.layer, st.layer)
    y = reshape(y_flat, Dim, Lcap, B)
    return y, (layer=st_layer,)
end

# A simplified Selective Scan (Mamba-like)
# y_t = SSM(x_t)
# We implement a basic version that is runnable in pure Julia.
# For full speed, this needs a custom kernel, but for prototype we use a loop.
struct SimplifiedMambaBlock{L1,L2,L3} <: Lux.AbstractLuxLayer
    in_proj::L1
    out_proj::L2
    # Hidden state dimension
    d_model::Int
    d_state::Int

    # Simple projections for B, C, Delta adaptation (simplified)
    adt_proj::L3
end

function SimplifiedMambaBlock(d_model::Int, d_state::Int=16)
    # Projects input to [x; z] (typical Mamba gated architecture)
    in_proj = Dense(d_model => d_model * 2)
    out_proj = Dense(d_model => d_model)

    # Project input to B, C, dt roughly
    # We simplify heavily: just predict params from x
    # dt (1), B (d_state), C (d_state)
    adt_proj = Dense(d_model => d_model + 2 * d_state)

    return SimplifiedMambaBlock(in_proj, out_proj, d_model, d_state, adt_proj)
end

Lux.initialparameters(rng::AbstractRNG, l::SimplifiedMambaBlock) = (
    in_proj=Lux.initialparameters(rng, l.in_proj),
    out_proj=Lux.initialparameters(rng, l.out_proj),
    adt_proj=Lux.initialparameters(rng, l.adt_proj),
    A=repeat(reshape(hippo_A_diag(l.d_state, Float32), 1, l.d_state), l.d_model, 1),
    D=ones(Float32, l.d_model)
)

Lux.initialstates(rng::AbstractRNG, l::SimplifiedMambaBlock) = (
    in_proj=Lux.initialstates(rng, l.in_proj),
    out_proj=Lux.initialstates(rng, l.out_proj),
    adt_proj=Lux.initialstates(rng, l.adt_proj)
)

# The scan function
# x: [D, L, B]
function mamba_scan(x, dt_raw, B_raw, C_raw, A, D)
    d_model, L, batch = size(x)
    d_state = size(A, 2)
    h = similar(x, d_model, d_state, batch)
    fill!(h, zero(eltype(h)))

    D2 = reshape(D, d_model, 1)
    A2 = reshape(A, d_model, d_state, 1)

    if L == 0
        return similar(x, d_model, 0, batch)
    end
    ys = similar(x, d_model, L, batch)

    dt_min = 1f-4
    dt_scale = 0.1f0
    for t in 1:L
        xt = view(x, :, t, :)
        dt_val = NNlib.softplus.(view(dt_raw, :, t, :)) .* dt_scale .+ dt_min
        dt = clamp.(dt_val, dt_min, 5f0)
        Bt = view(B_raw, :, t, :)
        Ct = view(C_raw, :, t, :)

        dt3 = reshape(dt, d_model, 1, batch)
        xt3 = reshape(xt, d_model, 1, batch)
        B3 = reshape(Bt, 1, d_state, batch)
        C3 = reshape(Ct, 1, d_state, batch)

        decay = exp.(A2 .* dt3)
        h = h .* decay .+ (B3 .* dt3) .* xt3

        y = dropdims(sum(h .* C3; dims=2); dims=2) .+ (D2 .* xt)
        @views ys[:, t, :] .= y
    end

    return ys
end

function mamba_scan_with_state(x, dt_raw, B_raw, C_raw, A, D)
    d_model, L, batch = size(x)
    d_state = size(A, 2)
    h = similar(x, d_model, d_state, batch)
    fill!(h, zero(eltype(h)))

    D2 = reshape(D, d_model, 1)
    A2 = reshape(A, d_model, d_state, 1)

    ys = similar(x, d_model, L, batch)
    hs = similar(x, d_model, d_state, batch, L)

    dt_min = 1f-4
    dt_scale = 0.1f0
    for t in 1:L
        xt = @view x[:, t, :]
        # Softplus is safe, but dt_scale * softplus can be large.
        # Clamp dt to avoid exp(A * dt) exploding or vanishing too hard.
        dt_val = NNlib.softplus.(@view dt_raw[:, t, :]) .* dt_scale .+ dt_min
        dt = clamp.(dt_val, dt_min, 5f0)
        Bt = @view B_raw[:, t, :]
        Ct = @view C_raw[:, t, :]

        dt3 = reshape(dt, d_model, 1, batch)
        xt3 = reshape(xt, d_model, 1, batch)
        B3 = reshape(Bt, 1, d_state, batch)
        C3 = reshape(Ct, 1, d_state, batch)

        decay = exp.(A2 .* dt3)
        h .= h .* decay .+ (B3 .* dt3) .* xt3

        y = dropdims(sum(h .* C3; dims=2); dims=2) .+ (D2 .* xt)
        @views ys[:, t, :] .= y
        @views hs[:, :, :, t] .= h
    end

    return ys, hs
end

function ChainRulesCore.rrule(::typeof(mamba_scan), x, dt_raw, B_raw, C_raw, A, D)
    y, hs = mamba_scan_with_state(x, dt_raw, B_raw, C_raw, A, D)

    function pullback(ȳ)
        if ȳ isa ChainRulesCore.AbstractZero
            return (ChainRulesCore.NoTangent(), ChainRulesCore.ZeroTangent(), ChainRulesCore.ZeroTangent(),
                ChainRulesCore.ZeroTangent(), ChainRulesCore.ZeroTangent(), ChainRulesCore.ZeroTangent(),
                ChainRulesCore.ZeroTangent())
        end

        d_model, L, batch = size(x)
        d_state = size(A, 2)

        dx = similar(x)
        fill!(dx, zero(eltype(dx)))
        ddt_raw = similar(dt_raw)
        fill!(ddt_raw, zero(eltype(ddt_raw)))
        dB_raw = similar(B_raw)
        fill!(dB_raw, zero(eltype(dB_raw)))
        dC_raw = similar(C_raw)
        fill!(dC_raw, zero(eltype(dC_raw)))
        dA = similar(A)
        fill!(dA, zero(eltype(dA)))
        dD = similar(D)
        fill!(dD, zero(eltype(dD)))

        D2 = reshape(D, d_model, 1)
        A2 = reshape(A, d_model, d_state, 1)

        h0 = similar(hs, d_model, d_state, batch)
        fill!(h0, zero(eltype(h0)))

        gh = similar(h0)
        fill!(gh, zero(eltype(gh)))
        gdecay = similar(h0)
        gudt = similar(x, d_model, batch)
        gdt = similar(x, d_model, batch)

        dt_min = 1f-4
        dt_scale = 0.1f0

        for t in L:-1:1
            xt = x[:, t, :]
            dt_raw_t = dt_raw[:, t, :]
            Bt = B_raw[:, t, :]
            Ct = C_raw[:, t, :]
            dt_val = NNlib.softplus.(dt_raw_t) .* dt_scale .+ dt_min
            dt = clamp.(dt_val, dt_min, 5f0)
            
            # Mask for dt gradient: 1 if not clamped, 0 if clamped
            dt_mask = (dt_val .>= dt_min) .& (dt_val .<= 5f0)

            dt3 = reshape(dt, d_model, 1, batch)
            xt3 = reshape(xt, d_model, 1, batch)
            B3 = reshape(Bt, 1, d_state, batch)
            C3 = reshape(Ct, 1, d_state, batch)

            h_prev = t == 1 ? h0 : hs[:, :, :, t - 1]
            h_t = hs[:, :, :, t]

            decay = exp.(A2 .* dt3)

            ȳt = ȳ[:, t, :]

            gh .+= reshape(ȳt, d_model, 1, batch) .* C3

            @views dC_raw[:, t, :] .+= dropdims(sum(h_t .* reshape(ȳt, d_model, 1, batch); dims=1); dims=1)
            dx[:, t, :] .+= ȳt .* D2
            dD .+= vec(dropdims(sum(ȳt .* xt; dims=2); dims=2))

            gdecay .= gh .* h_prev

            gudt .= dropdims(sum(gh .* (dt3 .* B3); dims=2); dims=2)
            dx[:, t, :] .+= gudt

            gdt .= dropdims(sum(gh .* (xt3 .* B3); dims=2); dims=2)
            @views dB_raw[:, t, :] .+= dropdims(sum(gh .* (xt3 .* dt3); dims=1); dims=1)

            gdt .+= dropdims(sum(gdecay .* (A2 .* decay); dims=2); dims=2)
            dA .+= dropdims(sum(gdecay .* (reshape(dt, d_model, 1, batch) .* decay); dims=3); dims=3)

            gdt_raw = gdt .* (dt_scale .* NNlib.sigmoid.(dt_raw_t))
            ddt_raw[:, t, :] .+= gdt_raw .* dt_mask

            gh .= gh .* decay
        end

        return (ChainRulesCore.NoTangent(), dx, ddt_raw, dB_raw, dC_raw, dA, dD)
    end

    return y, pullback
end

function (l::SimplifiedMambaBlock)(x, ps, st)
    # x: [D, L, B]
    xz = l.in_proj(x, ps.in_proj, st.in_proj)[1]
    d = l.d_model
    x_branch = copy(@view xz[1:d, :, :])
    z_branch = copy(@view xz[d+1:2d, :, :])

    adt, st_adt = l.adt_proj(x_branch, ps.adt_proj, st.adt_proj)
    d_state = l.d_state
    dt_raw = @view adt[1:d, :, :]
    B_raw = @view adt[d+1:d+d_state, :, :]
    C_raw = @view adt[d+d_state+1:d+2d_state, :, :]

    y_scan = mamba_scan(x_branch, dt_raw, B_raw, C_raw, ps.A, ps.D)
    y = y_scan .* NNlib.swish.(z_branch)

    out = l.out_proj(y, ps.out_proj, st.out_proj)[1]
    return out .+ x, (in_proj=st.in_proj, out_proj=st.out_proj, adt_proj=st_adt)
end

# --- 1. Mamba Encoder ---

struct MambaCompressor{L<:Lux.AbstractLuxLayer, H<:Lux.AbstractLuxLayer} <: Lux.AbstractLuxLayer
    embedding::Lux.Embedding
    layers::L
    head::H
    block_size::Int
    dim::Int
    pad_id::Int
    eos_id::Int
end

function MambaCompressor(vocab_size::Int, dim::Int, block_size::Int=8; pad_id::Int=1, eos_id::Int=2, mamba_d_state::Int=16)
    # 2 layers of "Simplified Mamba"
    layers = Chain(
        SimplifiedMambaBlock(dim, mamba_d_state),
        SimplifiedMambaBlock(dim, mamba_d_state),
        SimplifiedMambaBlock(dim, mamba_d_state),
        SimplifiedMambaBlock(dim, mamba_d_state)
    )
    # VAE Head: Project to 2 * dim (mu, logvar)
    head = Dense(dim => dim * 2)
    return MambaCompressor(Embedding(vocab_size => dim), layers, head, block_size, dim, pad_id, eos_id)
end

Lux.initialparameters(rng::AbstractRNG, m::MambaCompressor) = (
    embedding=Lux.initialparameters(rng, m.embedding),
    layers=Lux.initialparameters(rng, m.layers),
    head=Lux.initialparameters(rng, m.head)
)

Lux.initialstates(rng::AbstractRNG, m::MambaCompressor) = (
    embedding=Lux.initialstates(rng, m.embedding),
    layers=Lux.initialstates(rng, m.layers),
    head=Lux.initialstates(rng, m.head)
)

function (m::MambaCompressor)(x::AbstractMatrix{Int}, ps, st)
    # x: [L_seq, Batch] (Integers)
    # Embedding expects [L, B] -> [Dim, L, B]? No, usually [Dim, ...]

    # Lux Embedding: (in::AbstractVector) -> [out, in]
    # (in::AbstractMatrix) -> [out, in_1, in_2] => [Dim, L, B]

    seq_len_in = size(x, 1)
    B = size(x, 2)
    stride = max(m.block_size, 1)

    # Pad x to a multiple of block_size before passing to Mamba
    Lpad = (seq_len_in % stride == 0) ? seq_len_in : (seq_len_in + (stride - (seq_len_in % stride)))

    x_pad = if Lpad == seq_len_in
        x
    else
        pad_part = similar(x, Lpad - seq_len_in, B)
        Zygote.@ignore fill!(pad_part, m.pad_id)
        cat(x, pad_part; dims=1)
    end

    emb, st_emb = m.embedding(x_pad, ps.embedding, st.embedding)
    hidden, st_layers = m.layers(emb, ps.layers, st.layers)

    seq_len = Lpad
    Lcap = Lpad ÷ stride

    # hidden is already padded
    hidden4 = reshape(hidden, size(hidden, 1), stride, Lcap, B)
    # Always pool at `stride` since sequence is padded to full blocks
    pooled_slices = ntuple(c -> reshape(@view(hidden4[:, stride, c, :]), size(hidden4, 1), 1, B), Lcap)
    pooled = cat(pooled_slices...; dims=2)
    
    # helper for view preservation and batching
    s = size(pooled)
    # s is (Dim, Lcap, B)
    pooled_flat = reshape(pooled, s[1], s[2] * s[3])
    pooled_parent = parent(pooled_flat)
    if pooled_parent isa SubArray
        pooled_parent = parent(pooled_parent)
    end
    if pooled_parent isa CUDA.AbstractGPUArray
        pooled_flat = CUDA.CuArray(pooled_flat)
    end
    
    # Project to VAE params
    # pooled_flat: [Dim, Lcap * B]
    vae_out_flat, st_head = m.head(pooled_flat, ps.head, st.head)
    # vae_out_flat: [2*Dim, Lcap * B]
    
    # Reshape back to [2*Dim, Lcap, B]
    vae_out = reshape(vae_out_flat, :, s[2], s[3])
    
    # Split into mu and logvar
    # We assume dim 1 is 2*Dim
    mu = @view vae_out[1:m.dim, :, :]
    logvar = @view vae_out[m.dim+1:end, :, :]

    return (mu, logvar), (embedding=st_emb, layers=st_layers, head=st_head)
end
# --- 3. Transformer Reasoner (MHC Stage 2) ---

# Sinkhorn-Knopp Normalization for Doubly Stochastic Matrices
# M: [K, K] or [K, K, B]
function sinkhorn_knopp(M::AbstractArray{T}; iters::Int=5, eps::T=T(1e-6)) where T
    # M is shape [K, K] or [K, K, B]
    # Make strictly positive
    P = exp.(M)
    
    for _ in 1:iters
        # Row normalize
        row_sum = sum(P; dims=2)
        P = P ./ (row_sum .+ eps)
        # Col normalize
        col_sum = sum(P; dims=1)
        P = P ./ (col_sum .+ eps)
    end
    return P
end

# To support Sinkhorn in Zygote properly without custom adjoints for now, 
# the unrolled loop is differentiable.

struct SelfAttentionWrapper{M<:Lux.AbstractLuxLayer} <: Lux.AbstractLuxLayer
    mha::M
end
function SelfAttentionWrapper(dim::Int, heads::Int)
    return SelfAttentionWrapper(MultiHeadAttention(dim; nheads=heads))
end
Lux.initialparameters(rng::AbstractRNG, w::SelfAttentionWrapper) = (mha=Lux.initialparameters(rng, w.mha),)
Lux.initialstates(rng::AbstractRNG, w::SelfAttentionWrapper) = (mha=Lux.initialstates(rng, w.mha),)
function (w::SelfAttentionWrapper)(x, ps, st; mask=nothing)
    # x: [Dim, Seq, Batch]
    # MHA expects tuple (Q, K, V) or (Q, K, V, mask)
    # Returns (y, (attention_weights,)), st
    if mask === nothing
        (y, _), st_mha = w.mha((x, x, x), ps.mha, st.mha)
    else
        (y, _), st_mha = w.mha((x, x, x, mask), ps.mha, st.mha)
    end
    return y, (mha=st_mha,)
end

struct MHCTransformerBlock{N1, A, N2, F1, F2} <: Lux.AbstractLuxLayer
    norm1::N1
    attn::A
    norm2::N2
    ffn1::F1
    ffn2::F2
end

function MHCTransformerBlock(dim::Int, heads::Int)
    return MHCTransformerBlock(
        FeatureLayerNorm(dim),
        SelfAttentionWrapper(dim, heads),
        FeatureLayerNorm(dim),
        Dense(dim => 4dim, NNlib.gelu),
        Dense(4dim => dim)
    )
end

Lux.initialparameters(rng::AbstractRNG, m::MHCTransformerBlock) = (
    norm1=Lux.initialparameters(rng, m.norm1),
    attn=Lux.initialparameters(rng, m.attn),
    norm2=Lux.initialparameters(rng, m.norm2),
    ffn1=Lux.initialparameters(rng, m.ffn1),
    ffn2=Lux.initialparameters(rng, m.ffn2)
)

Lux.initialstates(rng::AbstractRNG, m::MHCTransformerBlock) = (
    norm1=Lux.initialstates(rng, m.norm1),
    attn=Lux.initialstates(rng, m.attn),
    norm2=Lux.initialstates(rng, m.norm2),
    ffn1=Lux.initialstates(rng, m.ffn1),
    ffn2=Lux.initialstates(rng, m.ffn2)
)

function (m::MHCTransformerBlock)(x, ps, st; mask=nothing)
    # Attention block with pre-norm and residual
    x_norm, st_norm1 = m.norm1(x, ps.norm1, st.norm1)
    y_attn, st_attn = m.attn(x_norm, ps.attn, st.attn; mask=mask)
    x = x .+ y_attn
    
    # FFN block with pre-norm and residual
    x_norm2, st_norm2 = m.norm2(x, ps.norm2, st.norm2)
    y_ffn1, st_ffn1 = m.ffn1(x_norm2, ps.ffn1, st.ffn1)
    y_ffn, st_ffn2 = m.ffn2(y_ffn1, ps.ffn2, st.ffn2)
    x = x .+ y_ffn
    
    return x, (norm1=st_norm1, attn=st_attn, norm2=st_norm2, ffn1=st_ffn1, ffn2=st_ffn2)
end

function k_drop_mask(x::AbstractArray, k_drop_threshold::Float32, k_drop_min::Int)
    scores = dropdims(Statistics.mean(abs2, x; dims=(1, 3, 4)); dims=(1, 3, 4))
    scores_vec = vec(scores)
    keep = scores_vec .>= k_drop_threshold
    if count(keep) < k_drop_min
        order = sortperm(scores_vec; rev=true)
        top = view(order, 1:k_drop_min)
        keep = in.(1:length(scores_vec), Ref(top))
    end
    return ifelse.(keep, one(eltype(x)), zero(eltype(x)))
end
Zygote.@nograd k_drop_mask

struct MHCBlock{T<:Lux.AbstractLuxLayer} <: Lux.AbstractLuxLayer
    transformer_block::T
    K_streams::Int
    k_drop_threshold::Float32
    k_drop_min::Int
end

function MHCBlock(dim::Int, K_streams::Int, heads::Int=8; k_drop_threshold::Float32=0f0, k_drop_min::Int=1)
    # A standard Pre-Norm Transformer Block with custom wrapper to support kwargs
    transformer_block = MHCTransformerBlock(dim, heads)
    kmin = max(1, min(k_drop_min, K_streams))
    return MHCBlock(transformer_block, K_streams, k_drop_threshold, kmin)
end

Lux.initialparameters(rng::AbstractRNG, m::MHCBlock) = (
    transformer_block=Lux.initialparameters(rng, m.transformer_block),
    # Learnable pre-mixing logits M_raw: [K, K]
    M_raw=randn(rng, Float32, m.K_streams, m.K_streams) .* 0.02f0
)

Lux.initialstates(rng::AbstractRNG, m::MHCBlock) = (
    transformer_block=Lux.initialstates(rng, m.transformer_block),
)

function (m::MHCBlock)(x, ps, st)
    # x: [Dim, K, L_seq, Batch]
    Dim, K, L, B = size(x)
    
    # 1. Manifold-Constrained Hyper-Connections (Mixing)
    if m.k_drop_threshold > 0f0 && K > m.k_drop_min
        mask_vals = k_drop_mask(x, m.k_drop_threshold, m.k_drop_min)
        mask = reshape(mask_vals, 1, K, 1, 1)
        x = x .* mask
    end

    # M: [K, K] -> Doubly Stochastic
    M = sinkhorn_knopp(ps.M_raw)
    
    # Mix across the K streams (dim 2)
    # x is [Dim, K, L, B], we want to multiply K dimension.
    # Permute to [K, Dim, L, B], reshape to [K, Dim * L * B]
    x_perm = permutedims(x, (2, 1, 3, 4))
    x_flat = reshape(x_perm, K, :)
    
    # M * x_flat -> [K, Dim * L * B]
    x_mixed_flat = M * x_flat
    
    # Reshape and permute back to [Dim, K, L, B]
    x_mixed_perm = reshape(x_mixed_flat, K, Dim, L, B)
    x_mixed = permutedims(x_mixed_perm, (2, 1, 3, 4))
    
    # Add Positional Encoding
    pos = sinusoidal_position_encoding(x_mixed, Dim, L)
    # pos is [Dim, L, 1]
    # x_mixed is [Dim, K, L, B], reshape pos to [Dim, 1, L, 1] for broadcasting
    pos_exp = reshape(pos, Dim, 1, L, 1)
    x_mixed = x_mixed .+ pos_exp

    # 2. Transformer Logic
    # Apply standard shared transformer to each stream independently (or treat K*B as batch)
    # transformer expects [Dim, L, Batch_effective]
    x_in_tf = reshape(x_mixed, Dim, L, K * B)
    
    # Create Causal Mask for L steps
    # Mask should be [L, L] (or [L, L, 1] depending on MultiHeadAttention implementation)
    # For Lux MultiHeadAttention, a 2D mask [L, L] is broadcasted across batch/heads.
    # We want a boolean lower triangular mask where true means keep, false means mask.
    causal_mask_cpu = LinearAlgebra.tril(ones(Bool, L, L))
    causal_mask = x_in_tf isa CUDA.AbstractGPUArray ? CUDA.CuArray(causal_mask_cpu) : causal_mask_cpu
    
    x_out_tf, st_tf = m.transformer_block(x_in_tf, ps.transformer_block, st.transformer_block; mask=causal_mask)
    
    # Reshape back to [Dim, K, L, B]
    x_out = reshape(x_out_tf, Dim, K, L, B)
    
    return x_out, (transformer_block=st_tf,)
end

struct MHCLatentReasoner{L<:Lux.AbstractLuxLayer} <: Lux.AbstractLuxLayer
    blocks::L
    K_streams::Int
    dim::Int
end

function MHCLatentReasoner(dim::Int, K_streams::Int=4, heads::Int=8, num_layers::Int=4; k_drop_threshold::Float32=0f0, k_drop_min::Int=1)
    # Initial Mixing / Expansion from 1 stream to K streams is handled outside or as a first step.
    # We assume input to Reasoner is already [Dim, K, L, B].
    blocks = Chain([MHCBlock(dim, K_streams, heads; k_drop_threshold=k_drop_threshold, k_drop_min=k_drop_min) for _ in 1:num_layers]...)
    return MHCLatentReasoner(blocks, K_streams, dim)
end

Lux.initialparameters(rng::AbstractRNG, r::MHCLatentReasoner) = (blocks=Lux.initialparameters(rng, r.blocks),)
Lux.initialstates(rng::AbstractRNG, r::MHCLatentReasoner) = (blocks=Lux.initialstates(rng, r.blocks),)

function (r::MHCLatentReasoner)(x, ps, st)
    # x: [Dim, K, Seq, Batch]
    y, st_blocks = r.blocks(x, ps.blocks, st.blocks)
    return y, (blocks=st_blocks,)
end

# --- Integration / Helper: Initial Mixing Layer ---
# Expands [Dim, L, B] to [Dim, K, L, B] using learned offsets (simplest unconditioned formulation)
struct InitialMixingLayer <: Lux.AbstractLuxLayer
    dim::Int
    K_streams::Int
end

Lux.initialparameters(rng::AbstractRNG, l::InitialMixingLayer) = (
    offsets=randn(rng, Float32, l.dim, l.K_streams) .* 0.02f0,
)
Lux.initialstates(rng::AbstractRNG, l::InitialMixingLayer) = NamedTuple()

function (l::InitialMixingLayer)(x, ps, st)
    # x: [Dim, L, B]
    Dim, L, B = size(x)
    
    # Add offsets to create K streams
    # x_exp: [Dim, 1, L, B]
    x_exp = reshape(x, Dim, 1, L, B)
    
    # offsets_exp: [Dim, K, 1, 1]
    offsets_exp = reshape(ps.offsets, Dim, l.K_streams, 1, 1)
    
    # Broadcasting plus: [Dim, K, L, B]
    x_k = x_exp .+ offsets_exp
    
    return x_k, st
end


# --- 3. Nano Decoder ---

struct NanoDecoder{E,C,P} <: Lux.AbstractLuxLayer
    embedding::E
    cell::C
    proj::P
    dim::Int
    block_size::Int
    eos_id::Int
end

function NanoDecoder(vocab_size::Int, dim::Int, block_size::Int=8; eos_id::Int=2)
    embedding = Embedding(vocab_size => dim)
    cell = GRUCell(dim => dim)
    proj = Dense(dim => vocab_size)
    return NanoDecoder(embedding, cell, proj, dim, block_size, eos_id)
end

Lux.initialparameters(rng::AbstractRNG, d::NanoDecoder) = (
    embedding=Lux.initialparameters(rng, d.embedding),
    cell=Lux.initialparameters(rng, d.cell),
    proj=Lux.initialparameters(rng, d.proj),
)

Lux.initialstates(rng::AbstractRNG, d::NanoDecoder) = (
    embedding=Lux.initialstates(rng, d.embedding),
    cell=Lux.initialstates(rng, d.cell),
    proj=Lux.initialstates(rng, d.proj),
)

function (d::NanoDecoder)(capsules, tokens::AbstractMatrix{Int}, ps, st; start_id::Int=d.eos_id, teacher_forcing::Bool=false)
    D, _, B = size(capsules)
    K = max(d.block_size, 1)

    L = size(tokens, 1)
    Lpad = (L % K == 0) ? L : (L + (K - (L % K)))
    Lcap2 = Lpad ÷ K

    tokens_pad = if Lpad == L
        tokens
    else
        pad_part = similar(tokens, Lpad - L, B)
        Zygote.@ignore fill!(pad_part, d.eos_id)
        cat(tokens, pad_part; dims=1)
    end

    t3 = reshape(tokens_pad, K, Lcap2, B)

    N = Lcap2 * B
    h = reshape(capsules, D, N)

    vocab_size = size(ps.proj.weight, 1)

    start_ids = similar(tokens_pad, N)
    Zygote.@ignore fill!(start_ids, start_id)

    st_emb = st.embedding
    st_cell = st.cell
    st_proj = st.proj

    logits_steps = ()
    prev_ids = start_ids
    for k in 1:K
        if teacher_forcing
            prev_ids = k == 1 ? start_ids : copy(reshape(@view(t3[k - 1, :, :]), N))
        end
        x_in, st_emb = d.embedding(prev_ids, ps.embedding, st_emb)
        (out, (h_new,)), st_cell = d.cell((x_in, (h,)), ps.cell, st_cell)
        h = h_new
        logits, st_proj = d.proj(out, ps.proj, st_proj)
        logits_steps = (logits_steps..., reshape(logits, vocab_size, 1, N))
        if !teacher_forcing && k < K
            logits_cpu = logits isa CUDA.AbstractGPUArray ? Array(logits) : logits
            prev_ids_cpu = [Int(findmax(@view logits_cpu[:, i])[2]) for i in 1:size(logits_cpu, 2)]
            prev_ids = logits isa CUDA.AbstractGPUArray ? CUDA.CuArray(prev_ids_cpu) : prev_ids_cpu
        end
    end

    out_buf = cat(logits_steps...; dims=2)
    out4 = reshape(out_buf, vocab_size, K, Lcap2, B)
    out_seq = reshape(out4, vocab_size, Lpad, B)

    return out_seq[:, 1:L, :], (embedding=st_emb, cell=st_cell, proj=st_proj)
end

# --- 3. Mamba Decoder ---

struct MambaDecoder{E,L,P} <: Lux.AbstractLuxLayer
    embedding::E
    layers::L
    proj::P
    dim::Int
    eos_id::Int
end

function MambaDecoder(vocab_size::Int, dim::Int; eos_id::Int=2, mamba_d_state::Int=16)
    embedding = Embedding(vocab_size => dim)
    # Isomorphic to Encoder: 4 layers of SimplifiedMambaBlock
    layers = Chain(
        SimplifiedMambaBlock(dim, mamba_d_state),
        SimplifiedMambaBlock(dim, mamba_d_state),
        SimplifiedMambaBlock(dim, mamba_d_state),
        SimplifiedMambaBlock(dim, mamba_d_state)
    )
    proj = Dense(dim => vocab_size)
    return MambaDecoder(embedding, layers, proj, dim, eos_id)
end

Lux.initialparameters(rng::AbstractRNG, d::MambaDecoder) = (
    embedding=Lux.initialparameters(rng, d.embedding),
    layers=Lux.initialparameters(rng, d.layers),
    proj=Lux.initialparameters(rng, d.proj)
)

Lux.initialstates(rng::AbstractRNG, d::MambaDecoder) = (
    embedding=Lux.initialstates(rng, d.embedding),
    layers=Lux.initialstates(rng, d.layers),
    proj=Lux.initialstates(rng, d.proj)
)

function sinusoidal_position_encoding(like, D::Int, L::Int)
    T = eltype(like)
    half = D ÷ 2

    positions_cpu = reshape(collect(0:(L - 1)), 1, L)
    div_idx_cpu = reshape(collect(0:(half - 1)), half, 1)

    positions = positions_cpu
    div_idx = div_idx_cpu
    if like isa CUDA.AbstractGPUArray
        positions = CUDA.CuArray(positions_cpu)
        div_idx = CUDA.CuArray(div_idx_cpu)
    end

    positions = T.(positions)
    div_idx = T.(div_idx)

    scale = -log(T(10000)) * (T(2) / T(D))
    div_term = exp.(div_idx .* scale)
    angles = div_term .* positions

    pe = similar(like, T, D, L)
    if half > 0
        sin_part = sin.(angles)
        cos_part = cos.(angles)
        @views pe[1:2:(2 * half), :] .= sin_part
        @views pe[2:2:(2 * half), :] .= cos_part
    end
    if 2 * half < D
        @views pe[(2 * half + 1):D, :] .= zero(T)
    end

    return reshape(pe, D, L, 1)
end

Zygote.@nograd sinusoidal_position_encoding

function (d::MambaDecoder)(capsules, ps, st; target_len::Int)
    # capsules: [Dim, 1, Batch] (or [Dim, N_capsules, Batch])
    # For Stage 1 AutoEncoder with Bucketing strategy, we typically have 1 capsule per sentence?
    # Or if we have N_capsules, we need to know how they map to target_len.
    # Assuming 1 capsule per "sequence" for now based on MambaCompressor global pooling.

    # Input: [Dim, 1, B]
    # Target: [Vocab, target_len, B]

    D, _, B = size(capsules)

    # Expand: [Dim, 1, B] -> [Dim, target_len, B]
    # We simply repeat the capsule vector.
    # Note: To allow Mamba to generate differentiated tokens, 
    # the internal state will evolve.
    # An option is to add positional embeddings here, but simple repetition is a good baseline.

    # Using Zygote-friendly repeat/broadcast
    # latents_expanded = repeat(capsules, 1, target_len, 1) # This might be inefficient or not supported well in all AD?
    Lcap = size(capsules, 2)
    rep = cld(target_len, Lcap)
    latents_expanded = repeat(capsules; inner=(1, rep, 1))
    latents_expanded = @view latents_expanded[:, 1:target_len, :]
    pos = sinusoidal_position_encoding(capsules, D, target_len)
    latents_expanded = latents_expanded .+ pos

    # Pass through Mamba layers
    features, st_layers = d.layers(latents_expanded, ps.layers, st.layers)

    # Project to Vocab
    # features: [Dim, L, B]
    logits, st_proj = d.proj(features, ps.proj, st.proj)
    # logits: [Vocab, L, B]

    return logits, (embedding=st.embedding, layers=st_layers, proj=st_proj)
end

# --- 4. HMTR Container ---

# --- 4. Stage 1 AutoEncoder Container ---

struct HMTR_Stage1_AutoEncoder{E,N,P,D} <: Lux.AbstractLuxLayer
    encoder::E
    norm::N
    predictor::P
    decoder::D
end

function HMTR_Stage1_AutoEncoder(vocab_size::Int, dim::Int=512; block_size::Int=8, pad_id::Int=1, eos_id::Int=2, mamba_d_state::Int=16)
    enc = MambaCompressor(vocab_size, dim, block_size; pad_id=pad_id, eos_id=eos_id, mamba_d_state=mamba_d_state)
    norm = FeatureLayerNorm(dim)
    pred = LatentPredictor(dim)
    dec = NanoDecoder(vocab_size, dim, block_size; eos_id=eos_id)
    return HMTR_Stage1_AutoEncoder(enc, norm, pred, dec)
end

Lux.initialparameters(rng::AbstractRNG, m::HMTR_Stage1_AutoEncoder) = (
    encoder=Lux.initialparameters(rng, m.encoder),
    norm=Lux.initialparameters(rng, m.norm),
    predictor=Lux.initialparameters(rng, m.predictor),
    decoder=Lux.initialparameters(rng, m.decoder)
)

Lux.initialstates(rng::AbstractRNG, m::HMTR_Stage1_AutoEncoder) = (
    encoder=Lux.initialstates(rng, m.encoder),
    norm=Lux.initialstates(rng, m.norm),
    predictor=Lux.initialstates(rng, m.predictor),
    decoder=Lux.initialstates(rng, m.decoder)
)

function (m::HMTR_Stage1_AutoEncoder)(x, ps, st; teacher_forcing::Bool=false)
    # x: [L, B] (Indices)
    L, B = size(x)

    # 1. Encode -> Capsules
    capsules_params, st_enc = m.encoder(x, ps.encoder, st.encoder)
    mu, logvar = capsules_params
    
    # Reparameterize (VAE)
    # Ideally should use randomness, here we use simple mean for inference if not specified, 
    # but for training we need noise.
    # We will let reparameterize handle it (it uses randn).
    z = reparameterize(mu, logvar)
    
    # 2. Normalize Capsules
    capsules_norm, st_norm = m.norm(z, ps.norm, st.norm)

    # 3. Latent Prediction (JEPA)
    # Predict "next" latent dynamics.
    # Note: predicting z_{t+1} from z_t. 
    # Current call returns z_pred corresponding to input z.
    # In loss function we will align z_pred[t] with z[t+1].
    z_pred, st_pred = m.predictor(capsules_norm, ps.predictor, st.predictor)
    
    # 4. Decode -> Text
    # Decoder reconstructs text from the *normalized* capsules
    logits, st_dec = m.decoder(capsules_norm, x, ps.decoder, st.decoder; start_id=m.encoder.eos_id, teacher_forcing=teacher_forcing)

    # Return structure:
    # logits: [Vocab, L, B]
    # mu, logvar: [Dim, N_capsules, B]
    # z: [Dim, N_capsules, B] (Sampled, normalized) - wait, capsules_norm is normalized z.
    # z_pred: [Dim, N_capsules, B] (Predicted next state)
    
    out = (; logits=logits, mu=mu, logvar=logvar, z=capsules_norm, z_pred=z_pred)
    new_st = (encoder=st_enc, norm=st_norm, predictor=st_pred, decoder=st_dec)
    
    return out, new_st
end

end # module Model
