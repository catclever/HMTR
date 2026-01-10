module Model

using Lux
using Random
using NNlib
using Zygote
using CUDA
using LuxCUDA

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
    Zygote.@ignore fill!(h, zero(eltype(h)))

    D2 = reshape(D, d_model, 1)
    A2 = reshape(A, d_model, d_state, 1)

    if L == 0
        return similar(x, d_model, 0, batch)
    end
    ys = Zygote.Buffer(x, d_model, L, batch)

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
        ys[:, t:t, :] = reshape(y, d_model, 1, batch)
    end

    return copy(ys)
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

    is_eos = x .== m.eos_id
    eos_cum = cumsum(Int.(is_eos); dims=1)
    mask_first_eos = (eos_cum .== 1) .& is_eos
    has_eos = any(is_eos; dims=1)

    is_nonpad = x .!= m.pad_id
    rev = reverse(is_nonpad; dims=1)
    rev_cum = cumsum(Int.(rev); dims=1)
    mask_rev = (rev_cum .== 1) .& rev
    mask_last = reverse(mask_rev; dims=1)

    mask = mask_first_eos .| ((.!has_eos) .& mask_last)
    mask_f = Float32.(mask)
    capsules = sum(hidden .* reshape(mask_f, 1, seq_len, B); dims=2)

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
    # Broadcast approach:
    # capsules is [D, 1, B]. We want [D, L, B].
    latents_expanded = capsules .+ zeros(Float32, D, target_len, B)

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

function HMTR_Stage1_AutoEncoder(vocab_size::Int, dim::Int=512; pad_id::Int=1, eos_id::Int=2, mamba_d_state::Int=16)
    # Encoder: block_size isn't strictly enforced in forward but kept in struct. 
    # We remove block_size dependency for MambaCompressor's ctor logic if it was just for struct.
    # But MambaCompressor defined in code has block_size. We pass 0 or a dummy if not used for striding.
    enc = MambaCompressor(vocab_size, dim, 0; pad_id=pad_id, eos_id=eos_id, mamba_d_state=mamba_d_state)
    norm = FeatureLayerNorm(dim)
    dec = MambaDecoder(vocab_size, dim; eos_id=eos_id, mamba_d_state=mamba_d_state)
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
    logits, st_dec = m.decoder(capsules_norm, ps.decoder, st.decoder; target_len=L)

    return logits, (encoder=st_enc, norm=st_norm, decoder=st_dec)
end

end # module Model
