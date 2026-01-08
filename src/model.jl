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
    scale = ones(Float32, l.dim, 1),
    bias = zeros(Float32, l.dim, 1),
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
struct SimplifiedMambaBlock{L1, L2, L3} <: Lux.AbstractLuxLayer
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
    in_proj = Lux.initialparameters(rng, l.in_proj),
    out_proj = Lux.initialparameters(rng, l.out_proj),
    adt_proj = Lux.initialparameters(rng, l.adt_proj),
    A = repeat(reshape(hippo_A_diag(l.d_state, Float32), 1, l.d_state), l.d_model, 1),
    D = ones(Float32, l.d_model)
)

Lux.initialstates(rng::AbstractRNG, l::SimplifiedMambaBlock) = (
    in_proj = Lux.initialstates(rng, l.in_proj),
    out_proj = Lux.initialstates(rng, l.out_proj),
    adt_proj = Lux.initialstates(rng, l.adt_proj)
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

struct MambaCompressor{L <: Lux.AbstractLuxLayer} <: Lux.AbstractLuxLayer
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
        SimplifiedMambaBlock(dim, mamba_d_state)
    )
    return MambaCompressor(Embedding(vocab_size => dim), layers, block_size, dim, pad_id, eos_id)
end

Lux.initialparameters(rng::AbstractRNG, m::MambaCompressor) = (
    embedding = Lux.initialparameters(rng, m.embedding),
    layers = Lux.initialparameters(rng, m.layers)
)

Lux.initialstates(rng::AbstractRNG, m::MambaCompressor) = (
    embedding = Lux.initialstates(rng, m.embedding),
    layers = Lux.initialstates(rng, m.layers)
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

struct LatentReasoner{T <: Lux.AbstractLuxLayer} <: Lux.AbstractLuxLayer
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
                 Chain(FeatureLayerNorm(dim), Dense(dim=>4dim, gelu), Dense(4dim=>dim)),
                 +
            )
        ) for _ in 1:layers
    ]
    
    return LatentReasoner(Chain(blocks...))
end

Lux.initialparameters(rng::AbstractRNG, r::LatentReasoner) = (
    transformer = Lux.initialparameters(rng, r.transformer),
)

Lux.initialstates(rng::AbstractRNG, r::LatentReasoner) = (
    transformer = Lux.initialstates(rng, r.transformer),
)

function (r::LatentReasoner)(x, ps, st)
    # x: [Dim, Seq, Batch]
    return r.transformer(x, ps.transformer, st.transformer)
end

# --- 3. Nano-RNN Decoder ---

struct NanoDecoder{E, R, P} <: Lux.AbstractLuxLayer
    embedding::E
    cell::R
    proj::P
    block_size::Int
    dim::Int
end

function NanoDecoder(dim::Int, vocab_size::Int, block_size::Int)
    embedding = Embedding(vocab_size => dim)
    cell = GRUCell(dim * 2 => dim)
    proj = Dense(dim => vocab_size)
    return NanoDecoder(embedding, cell, proj, block_size, dim)
end

Lux.initialparameters(rng::AbstractRNG, d::NanoDecoder) = (
    embedding = Lux.initialparameters(rng, d.embedding),
    cell = Lux.initialparameters(rng, d.cell),
    proj = Lux.initialparameters(rng, d.proj)
)

Lux.initialstates(rng::AbstractRNG, d::NanoDecoder) = (
    embedding = Lux.initialstates(rng, d.embedding),
    cell = Lux.initialstates(rng, d.cell),
    proj = Lux.initialstates(rng, d.proj)
)

function (d::NanoDecoder)(latents, ps, st; start_id::Int=2)
    # latents: [Dim, Seq_Capsules, Batch]
    # Output: [Vocab, Total_Tokens, Batch]
    
    D, L_cap, B = size(latents)
    K = d.block_size
    
    # Reshape latents to [D, L_cap * B]
    latents_flat = copy(reshape(latents, D, :)) # [D, N_capsules]
    
    # Initialize RNN state with latents
    h = latents_flat
    
    # Autoregressive generation
    # We feed start_id (usually EOS) to begin, then feed the predicted token back.
    
    vocab_size = size(ps.proj.weight, 1)
    N_capsules = size(latents_flat, 2)
    
    # Output buffer
    # Note: Zygote.Buffer is needed if we want to be AD-friendly, 
    # but argmax is non-differentiable anyway. 
    # If this is strictly for inference, Array is fine.
    # We'll use Zygote.Buffer to be safe with Lux conventions, 
    # though gradients won't flow through argmax.
    out_buf = Zygote.Buffer(similar(latents_flat, vocab_size, K, N_capsules))
    
    st_emb = st.embedding
    st_cell = st.cell
    st_proj = st.proj
    
    # Initial input: start_id
    # We need to create an input vector of start_ids
    # Using the embedding layer to get the vector.
    
    curr_ids = fill(start_id, N_capsules)
    # Move to same device as parameters if possible? 
    # We can infer device from ps.proj.weight
    # But for simplicity, we rely on Lux handling array types. 
    # If ps is on GPU, we need curr_ids on GPU.
    # We can use a helper or just hope the embedding layer handles CPU indices -> GPU output?
    # Lux Embedding usually handles CPU indices.
    
    # For GPU compatibility, we might need to make sure curr_ids matches device.
    # But we don't have easy access to device here.
    # However, standard Lux Embedding works with Integer arrays (CPU or GPU).
    
    x_in, st_emb = d.embedding(curr_ids, ps.embedding, st_emb)
    
    for k in 1:K
        rnn_input = vcat(x_in, latents_flat)
        (out, (h_new,)), st_cell = d.cell((rnn_input, (h,)), ps.cell, st_cell)
        h = h_new
        
        logits, st_proj = d.proj(out, ps.proj, st_proj)
        out_buf[:, k, :] = logits
        
        if k < K
            # Greedy decoding: argmax
            # logits: [Vocab, N_capsules]
            
            # We need to find argmax. 
            # Note: For AD, this breaks the graph. This is strictly for inference/eval.
            
            # We use dropdims(argmax(...)) logic
            # argmax returns CartesianIndex.
            
            # Helper to get indices. 
            # We need to handle CUDA arrays properly if on GPU.
            # CUDA.argmax works? Yes.
            
            # However, we need to extract the integer index.
            # map(c -> c[1], argmax(logits, dims=1))
            
            # To stay generic:
            indices = argmax(logits; dims=1)
            # indices is [1, N_capsules] of CartesianIndex{2}
            
            next_ids = vec(map(idx -> idx[1], indices))
            
            # Feed back
            x_in, st_emb = d.embedding(next_ids, ps.embedding, st_emb)
        end
    end
    
    out_flat = copy(out_buf)
    
    # Reshape back to [Vocab, Total_Seq, B]
    # out_flat: [Vocab, K, L*B]
    # We want to merge K and L such that they are sequential blocks.
    # [Vocab, K, L, B] -> [Vocab, K*L, B]
    
    out_reshaped = reshape(out_flat, size(out_flat, 1), K, L_cap, B)
    final_out = reshape(out_reshaped, size(out_reshaped, 1), :, B)
    
    return final_out, (embedding=st_emb, cell=st_cell, proj=st_proj)
end

function (d::NanoDecoder)(latents, targets::AbstractMatrix{Int}, ps, st; start_id::Int)
    D, L_cap, B = size(latents)
    K = d.block_size

    L_total = size(targets, 1)
    if L_total != K * L_cap
        throw(ArgumentError("targets must have length K*L_cap. Got L_total=\$L_total, K=\$K, L_cap=\$L_cap"))
    end

    latents_flat = copy(reshape(latents, D, :))
    targets_steps = reshape(targets, K, L_cap, B)
    targets_flat = reshape(targets_steps, K, :)

    N_capsules = size(latents_flat, 2)
    h = latents_flat

    vocab_size = size(ps.proj.weight, 1)
    out_buf = Zygote.Buffer(similar(latents_flat, vocab_size, K, N_capsules))

    st_emb = st.embedding
    st_cell = st.cell
    st_proj = st.proj

    start_ids = fill(start_id, N_capsules)
    x_in, st_emb = d.embedding(start_ids, ps.embedding, st_emb)

    for k in 1:K
        rnn_input = vcat(x_in, latents_flat)
        (out, (h_new,)), st_cell = d.cell((rnn_input, (h,)), ps.cell, st_cell)
        h = h_new
        logits, st_proj = d.proj(out, ps.proj, st_proj)
        out_buf[:, k, :] = logits
        if k < K
            x_in, st_emb = d.embedding(view(targets_flat, k, :), ps.embedding, st_emb)
        end
    end

    out_flat = copy(out_buf)
    out_reshaped = reshape(out_flat, size(out_flat, 1), K, L_cap, B)
    final_out = reshape(out_reshaped, size(out_reshaped, 1), :, B)
    return final_out, (embedding=st_emb, cell=st_cell, proj=st_proj)
end

# --- 4. HMTR Container ---

# --- 4. Stage 1 AutoEncoder Container ---

struct HMTR_Stage1_AutoEncoder{E, N, D} <: Lux.AbstractLuxLayer
    encoder::E
    norm::N
    decoder::D
end

function HMTR_Stage1_AutoEncoder(vocab_size::Int, dim::Int=512, block_size::Int=8; pad_id::Int=1, eos_id::Int=2, mamba_d_state::Int=16)
    enc = MambaCompressor(vocab_size, dim, block_size; pad_id=pad_id, eos_id=eos_id, mamba_d_state=mamba_d_state)
    norm = FeatureLayerNorm(dim)
    dec = NanoDecoder(dim, vocab_size, block_size)
    return HMTR_Stage1_AutoEncoder(enc, norm, dec)
end

Lux.initialparameters(rng::AbstractRNG, m::HMTR_Stage1_AutoEncoder) = (
    encoder = Lux.initialparameters(rng, m.encoder),
    norm = Lux.initialparameters(rng, m.norm),
    decoder = Lux.initialparameters(rng, m.decoder)
)

Lux.initialstates(rng::AbstractRNG, m::HMTR_Stage1_AutoEncoder) = (
    encoder = Lux.initialstates(rng, m.encoder),
    norm = Lux.initialstates(rng, m.norm),
    decoder = Lux.initialstates(rng, m.decoder)
)

function (m::HMTR_Stage1_AutoEncoder)(x, ps, st; start_id=nothing)
    capsules, st_enc = m.encoder(x, ps.encoder, st.encoder)
    capsules_norm, st_norm = m.norm(capsules, ps.norm, st.norm)
    
    sid = start_id === nothing ? m.encoder.eos_id : start_id
    logits, st_dec = m.decoder(capsules_norm, ps.decoder, st.decoder; start_id=sid)
    
    return logits, (encoder=st_enc, norm=st_norm, decoder=st_dec)
end

end # module Model
