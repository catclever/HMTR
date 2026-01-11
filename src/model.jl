module Model

using Lux
using Random
using NNlib
using Zygote
using CUDA
using LuxCUDA
import ChainRulesCore
import ChainRulesCore

export FeatureLayerNorm, SimplifiedMambaBlock, MambaCompressor, LatentReasoner, NanoDecoder, HMTR_Stage1_AutoEncoder

hippo_A_diag(d_state::Int, ::Type{T}=Float32) where {T<:AbstractFloat} = -(T.(collect(1:d_state)))

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
        dt = NNlib.softplus.(view(dt_raw, :, t, :)) .* dt_scale .+ dt_min
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
        dt = NNlib.softplus.(@view dt_raw[:, t, :]) .* dt_scale .+ dt_min
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
            dt = NNlib.softplus.(dt_raw_t) .* dt_scale .+ dt_min

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

            ddt_raw[:, t, :] .+= gdt .* (dt_scale .* NNlib.sigmoid.(dt_raw_t))

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

struct MambaCompressor{L<:Lux.AbstractLuxLayer} <: Lux.AbstractLuxLayer
    embedding::Lux.Embedding
    layers::L
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
    return MambaCompressor(Embedding(vocab_size => dim), layers, block_size, dim, pad_id, eos_id)
end

Lux.initialparameters(rng::AbstractRNG, m::MambaCompressor) = (
    embedding=Lux.initialparameters(rng, m.embedding),
    layers=Lux.initialparameters(rng, m.layers)
)

Lux.initialstates(rng::AbstractRNG, m::MambaCompressor) = (
    embedding=Lux.initialstates(rng, m.embedding),
    layers=Lux.initialstates(rng, m.layers)
)

function (m::MambaCompressor)(x::AbstractMatrix{Int}, ps, st)
    # x: [L_seq, Batch] (Integers)
    # Embedding expects [L, B] -> [Dim, L, B]? No, usually [Dim, ...]

    # Lux Embedding: (in::AbstractVector) -> [out, in]
    # (in::AbstractMatrix) -> [out, in_1, in_2] => [Dim, L, B]

    emb, st_emb = m.embedding(x, ps.embedding, st.embedding)
    # emb: [Dim, Seq, Batch]

    hidden, st_layers = m.layers(emb, ps.layers, st.layers)

    seq_len = size(hidden, 2)
    B = size(hidden, 3)

    stride = max(m.block_size, 1)
    Lpad = (seq_len % stride == 0) ? seq_len : (seq_len + (stride - (seq_len % stride)))
    Lcap = Lpad ÷ stride

    hidden_pad = if Lpad == seq_len
        hidden
    else
        pad_part = similar(hidden, size(hidden, 1), Lpad - seq_len, B)
        Zygote.@ignore fill!(pad_part, zero(eltype(pad_part)))
        cat(hidden, pad_part; dims=2)
    end

    hidden4 = reshape(hidden_pad, size(hidden_pad, 1), stride, Lcap, B)
    capsules = @view hidden4[:, stride, :, :]

    return capsules, (embedding=st_emb, layers=st_layers)
end

# --- 2. Transformer Reasoner ---

struct LatentReasoner{T<:Lux.AbstractLuxLayer} <: Lux.AbstractLuxLayer
    transformer::T
end

function LatentReasoner(dim::Int, heads::Int=8, layers::Int=4)
    # Standard Transformer Encoder
    # Input: [Dim, Seq_Capsules, Batch]

    # We construct simplified Transformer blocks
    # Lux doesn't have a turn-key "TransformerEncoder" with stacked blocks exported easily in all versions.
    # We make a simple chain of blocks.

    blocks = [
        Chain(
            SkipConnection(
                Chain(FeatureLayerNorm(dim), MultiHeadAttention(dim, nheads=heads)),
                +
            ),
            SkipConnection(
                Chain(FeatureLayerNorm(dim), Dense(dim => 4dim, gelu), Dense(4dim => dim)),
                +
            )
        ) for _ in 1:layers
    ]

    return LatentReasoner(Chain(blocks...))
end

Lux.initialparameters(rng::AbstractRNG, r::LatentReasoner) = (
    transformer=Lux.initialparameters(rng, r.transformer),
)

Lux.initialstates(rng::AbstractRNG, r::LatentReasoner) = (
    transformer=Lux.initialstates(rng, r.transformer),
)

function (r::LatentReasoner)(x, ps, st)
    # x: [Dim, Seq, Batch]
    return r.transformer(x, ps.transformer, st.transformer)
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

function (d::NanoDecoder)(capsules, tokens::AbstractMatrix{Int}, ps, st; start_id::Int=d.eos_id)
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
    for k in 1:K
        prev_ids = k == 1 ? start_ids : copy(reshape(@view(t3[k - 1, :, :]), N))
        x_in, st_emb = d.embedding(prev_ids, ps.embedding, st_emb)
        (out, (h_new,)), st_cell = d.cell((x_in, (h,)), ps.cell, st_cell)
        h = h_new
        logits, st_proj = d.proj(out, ps.proj, st_proj)
        logits_steps = (logits_steps..., reshape(logits, vocab_size, 1, N))
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
    if like isa Union{CUDA.CuArray, CUDA.CuDeviceArray}
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

struct HMTR_Stage1_AutoEncoder{E,N,D} <: Lux.AbstractLuxLayer
    encoder::E
    norm::N
    decoder::D
end

function HMTR_Stage1_AutoEncoder(vocab_size::Int, dim::Int=512; block_size::Int=8, pad_id::Int=1, eos_id::Int=2, mamba_d_state::Int=16)
    # Encoder: block_size isn't strictly enforced in forward but kept in struct. 
    # We remove block_size dependency for MambaCompressor's ctor logic if it was just for struct.
    # But MambaCompressor defined in code has block_size. We pass 0 or a dummy if not used for striding.
    enc = MambaCompressor(vocab_size, dim, block_size; pad_id=pad_id, eos_id=eos_id, mamba_d_state=mamba_d_state)
    norm = FeatureLayerNorm(dim)
    dec = NanoDecoder(vocab_size, dim, block_size; eos_id=eos_id)
    return HMTR_Stage1_AutoEncoder(enc, norm, dec)
end

Lux.initialparameters(rng::AbstractRNG, m::HMTR_Stage1_AutoEncoder) = (
    encoder=Lux.initialparameters(rng, m.encoder),
    norm=Lux.initialparameters(rng, m.norm),
    decoder=Lux.initialparameters(rng, m.decoder)
)

Lux.initialstates(rng::AbstractRNG, m::HMTR_Stage1_AutoEncoder) = (
    encoder=Lux.initialstates(rng, m.encoder),
    norm=Lux.initialstates(rng, m.norm),
    decoder=Lux.initialstates(rng, m.decoder)
)

function (m::HMTR_Stage1_AutoEncoder)(x, ps, st; kwargs...)
    # x: [Dim (if embedded) or Indices, L, B]
    # Note: MambaCompressor expects indices [L, B]? Let's check.
    # MambaCompressor line 173: (x::AbstractMatrix{Int}, ...) -> [L_seq, Batch]
    # BUT line 181: m.embedding(x...) suggests x is indices.

    # Get target length from input x
    L, B = size(x)

    capsules, st_enc = m.encoder(x, ps.encoder, st.encoder)
    capsules_norm, st_norm = m.norm(capsules, ps.norm, st.norm)

    # Pass target_len = L to decoder
    logits, st_dec = m.decoder(capsules_norm, x, ps.decoder, st.decoder; start_id=m.encoder.eos_id)

    return logits, (encoder=st_enc, norm=st_norm, decoder=st_dec)
end

end # module Model
