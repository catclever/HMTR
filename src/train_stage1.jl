module TrainStage1

using Lux
using Random
using JLD2
using Optimisers
using Zygote
using Printf
using NNlib
using CUDA
using LuxCUDA
import ChainRulesCore
using Dates
using Statistics
using ..Utils
using ..Model

Zygote.@nograd CUDA.driver_version
Zygote.@nograd CUDA.task_local_state!
Zygote.@nograd CUDA.isvalid
Zygote.@nograd CUDA.maybe_collect
Zygote.@nograd CUDA.randn
Zygote.@nograd CUDA.rand
Zygote.@nograd CUDA.CURAND.default_rng

export train_stage1

const PROJECT_ROOT = normpath(joinpath(@__DIR__, ".."))
const DATA_DIR = get(ENV, "DATA_DIR", joinpath(PROJECT_ROOT, "data"))
const DATA_FILE = get(ENV, "DATA_FILE", joinpath(DATA_DIR, "processed.jld2"))
const META_FILE = get(ENV, "META_FILE", "")
const CHECKPOINT_DIR = get(ENV, "CHECKPOINT_DIR", joinpath(PROJECT_ROOT, "checkpoints"))
const CHECKPOINT_PREFIX = get(ENV, "CHECKPOINT_PREFIX", "ckpt_stage1")
const EPOCHS = parse(Int, get(ENV, "EPOCHS", "5"))
const BATCH_SIZE = parse(Int, get(ENV, "BATCH_SIZE", "32"))
const LR = parse(Float64, get(ENV, "LR", "1e-3"))
const MAX_BATCHES = parse(Int, get(ENV, "MAX_BATCHES", "0"))
const SAVE_EVERY = parse(Int, get(ENV, "SAVE_EVERY", "0"))
const ADD_TIMESTAMP = parse(Int, get(ENV, "ADD_TIMESTAMP", "1"))
const GRAD_CLIP_NORM = parse(Float64, get(ENV, "GRAD_CLIP_NORM", "5.0"))
const LOSS_SPIKE_THRESHOLD = parse(Float64, get(ENV, "LOSS_SPIKE_THRESHOLD", "10.0"))
const SKIP_ON_SPIKE = parse(Int, get(ENV, "SKIP_ON_SPIKE", "1"))
const WARMUP_STEPS = parse(Int, get(ENV, "WARMUP_STEPS", "500"))
const KL_WEIGHT = parse(Float64, get(ENV, "KL_WEIGHT", "0.0001"))
const PRED_WEIGHT = parse(Float64, get(ENV, "PRED_WEIGHT", "1.0"))

const PRETRAIN_EMB_FILE = get(ENV, "PRETRAIN_EMB_FILE", "")
const INSPECT_DATA = parse(Int, get(ENV, "INSPECT_DATA", "0"))
const INSPECT_N = parse(Int, get(ENV, "INSPECT_N", "3"))
const INSPECT_SEED = parse(Int, get(ENV, "INSPECT_SEED", "42"))
const MODEL_DIM = parse(Int, get(ENV, "MODEL_DIM", "256"))
const MAMBA_D_STATE = parse(Int, get(ENV, "MAMBA_D_STATE", "16"))
const DTYPE = get(ENV, "DTYPE", "fp32")
const ENCODER_DTYPE = get(ENV, "ENCODER_DTYPE", "")
const NORM_DTYPE = get(ENV, "NORM_DTYPE", "")
const DECODER_DTYPE = get(ENV, "DECODER_DTYPE", "")
const RESUME_CKPT = get(ENV, "RESUME_CKPT", "")
const SEQ_LEN = parse(Int, get(ENV, "SEQ_LEN", "1024"))

function parse_ckpt_epoch_step(path::AbstractString)
    base = basename(path)
    m = match(r"_epoch(\d+)(?:_step(\d+))?\.jld2$", base)
    if m === nothing
        return 0, 0
    end
    epoch = parse(Int, m.captures[1])
    step = m.captures[2] === nothing ? 0 : parse(Int, m.captures[2])
    return epoch, step
end

function infer_ckpt_prefix(path::AbstractString)
    base = basename(path)
    if endswith(base, ".jld2")
        base = base[1:end-5]
    end
    m = match(r"^(.*)_epoch\d+(?:_step\d+)?$", base)
    return m === nothing ? base : m.captures[1]
end


function resolve_config(cli::Dict{Symbol,Any})
    data_file = get(cli, :data_file, DATA_FILE)
    if !isabspath(data_file)
        if !isfile(data_file) && isfile(joinpath(DATA_DIR, data_file))
            data_file = joinpath(DATA_DIR, data_file)
        end
    end
    meta_file = string(get(cli, :meta_file, META_FILE))
    if isempty(meta_file)
        if endswith(data_file, ".jld2")
            meta_file = data_file[1:end-5] * "_meta.jld2"
        else
            meta_file = data_file * "_meta.jld2"
        end
    end
    if !isabspath(meta_file) && !isempty(meta_file)
        if !isfile(meta_file) && isfile(joinpath(DATA_DIR, meta_file))
            meta_file = joinpath(DATA_DIR, meta_file)
        end
    end
    checkpoint_dir = get(cli, :checkpoint_dir, CHECKPOINT_DIR)
    checkpoint_prefix = get(cli, :checkpoint_prefix, CHECKPOINT_PREFIX)
    epochs = parse(Int, string(get(cli, :epochs, EPOCHS)))
    batch_size = parse(Int, string(get(cli, :batch_size, BATCH_SIZE)))
    dim = parse(Int, string(get(cli, :dim, MODEL_DIM)))
    mamba_d_state = parse(Int, string(get(cli, :mamba_d_state, MAMBA_D_STATE)))
    lr = parse(Float64, string(get(cli, :lr, LR)))
    max_batches = parse(Int, string(get(cli, :max_batches, MAX_BATCHES)))
    save_every = parse(Int, string(get(cli, :save_every, SAVE_EVERY)))

    function parse_bool(key, env_default_int)
        val = string(get(cli, key, env_default_int))
        return val == "true" || (tryparse(Int, val) !== nothing && parse(Int, val) != 0)
    end

    add_timestamp = parse_bool(:add_timestamp, ADD_TIMESTAMP) || (string(get(cli, :timestamp, "false")) == "true")

    grad_clip_norm = parse(Float64, string(get(cli, :grad_clip_norm, GRAD_CLIP_NORM)))
    loss_spike_threshold = parse(Float64, string(get(cli, :loss_spike_threshold, LOSS_SPIKE_THRESHOLD)))

    skip_on_spike = parse_bool(:skip_on_spike, SKIP_ON_SPIKE) || (string(get(cli, :skip_spike, "false")) == "true")

    pretrain_emb_file = get(cli, :pretrain_emb_file, PRETRAIN_EMB_FILE)

    inspect_data = parse_bool(:inspect_data, INSPECT_DATA) || (string(get(cli, :inspect, "false")) == "true")

    inspect_n = parse(Int, string(get(cli, :inspect_n, INSPECT_N)))
    warmup_steps = parse(Int, string(get(cli, :warmup_steps, WARMUP_STEPS)))
    kl_weight = parse(Float64, string(get(cli, :kl_weight, KL_WEIGHT)))
    pred_weight = parse(Float64, string(get(cli, :pred_weight, PRED_WEIGHT)))
    inspect_seed = parse(Int, string(get(cli, :inspect_seed, INSPECT_SEED)))
    dtype = string(get(cli, :dtype, DTYPE))
    encoder_dtype = string(get(cli, :encoder_dtype, ENCODER_DTYPE))
    norm_dtype = string(get(cli, :norm_dtype, NORM_DTYPE))
    decoder_dtype = string(get(cli, :decoder_dtype, DECODER_DTYPE))
    seq_len = parse(Int, string(get(cli, :seq_len, SEQ_LEN)))

    resume_ckpt = string(get(cli, :resume_ckpt, get(cli, :resume, get(cli, :resume_from, RESUME_CKPT))))
    if resume_ckpt == "true"
        resume_ckpt = ""
    end

    # Smart path resolution:
    # 1. Check if path exists as-is (absolute or relative to CWD)
    # 2. If not, try relative to checkpoint_dir
    if !isempty(resume_ckpt) && !isfile(resume_ckpt)
        alt_path = joinpath(checkpoint_dir, basename(resume_ckpt))
        if isfile(alt_path)
            resume_ckpt = alt_path
        elseif isfile(joinpath(checkpoint_dir, resume_ckpt))
            resume_ckpt = joinpath(checkpoint_dir, resume_ckpt)
        end
    end

    resume_epoch, resume_step = isempty(resume_ckpt) ? (0, 0) : parse_ckpt_epoch_step(resume_ckpt)

    prefix_explicit = haskey(cli, :checkpoint_prefix) || haskey(ENV, "CHECKPOINT_PREFIX")
    timestamp_explicit = haskey(cli, :add_timestamp) || haskey(cli, :timestamp) || haskey(ENV, "ADD_TIMESTAMP")
    if !isempty(resume_ckpt) && !prefix_explicit
        checkpoint_prefix = infer_ckpt_prefix(resume_ckpt)
        prefix_explicit = true
    end
    if !isempty(resume_ckpt) && !timestamp_explicit
        add_timestamp = false
    end

    if dim != MODEL_DIM && !occursin("_d$(dim)", checkpoint_prefix)
        checkpoint_prefix = "$(checkpoint_prefix)_d$(dim)"
    end
    if mamba_d_state != MAMBA_D_STATE && !occursin("_ds$(mamba_d_state)", checkpoint_prefix)
        checkpoint_prefix = "$(checkpoint_prefix)_ds$(mamba_d_state)"
    end

    if !prefix_explicit || add_timestamp
        ts = Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
        checkpoint_prefix = "$(checkpoint_prefix)_$(ts)"
    end

    return (;
        data_file,
        meta_file,
        checkpoint_dir,
        checkpoint_prefix,
        epochs,
        batch_size,
        dim,
        mamba_d_state,
        lr,
        max_batches,
        save_every,
        grad_clip_norm,
        loss_spike_threshold,
        skip_on_spike,
        pretrain_emb_file,
        resume_ckpt,
        resume_epoch,
        resume_step,
        inspect_data,
        inspect_n,
        inspect_seed,
        dtype,
        encoder_dtype,
        norm_dtype,
        decoder_dtype,
        warmup_steps,
        kl_weight,
        pred_weight,
        seq_len,
    )
end


function tree_sumabs2(x)
    if x === nothing
        return 0f0
    elseif x isa NamedTuple
        s = 0f0
        for v in values(x)
            s += tree_sumabs2(v)
        end
        return s
    elseif x isa Tuple
        s = 0f0
        for v in x
            s += tree_sumabs2(v)
        end
        return s
    elseif x isa AbstractArray
        return Float32(sum(abs2, x))
    elseif x isa Number
        return Float32(abs2(x))
    else
        return 0f0
    end
end

function tree_map(f, x)
    if x === nothing
        return nothing
    elseif x isa NamedTuple
        ks = keys(x)
        vs = map(v -> tree_map(f, v), Tuple(values(x)))
        return NamedTuple{ks}(vs)
    elseif x isa Tuple
        return map(v -> tree_map(f, v), x)
    else
        return f(x)
    end
end

function parse_dtype(s::AbstractString)
    v = lowercase(strip(String(s)))
    if v in ("", "default")
        return nothing
    elseif v in ("fp32", "float32", "f32")
        return Float32
    elseif v in ("fp16", "float16", "f16")
        return Float16
    elseif v in ("bf16", "bfloat16")
        return Core.BFloat16
    else
        error("Unknown dtype=$(s). Supported: fp32, fp16, bf16")
    end
end

function cast_floats(x, T::Type)
    return tree_map(v -> (v isa AbstractArray && eltype(v) <: AbstractFloat && eltype(v) != T) ? T.(v) : v, x)
end

function apply_precision(ps, st, cfg)
    T_global = something(parse_dtype(cfg.dtype), Float32)
    T_enc = something(parse_dtype(cfg.encoder_dtype), T_global)
    T_norm = something(parse_dtype(cfg.norm_dtype), T_global)
    T_dec = something(parse_dtype(cfg.decoder_dtype), T_global)

    ps2 = merge(ps, (;
        encoder=cast_floats(ps.encoder, T_enc),
        norm=cast_floats(ps.norm, T_norm),
        decoder=cast_floats(ps.decoder, T_dec),
    ))

    st2 = st
    if haskey(st, :encoder) && haskey(st, :norm) && haskey(st, :decoder)
        st2 = merge(st, (;
            encoder=cast_floats(st.encoder, T_enc),
            norm=cast_floats(st.norm, T_norm),
            decoder=cast_floats(st.decoder, T_dec),
        ))
    end

    return ps2, st2, (; global_dtype=T_global, encoder=T_enc, norm=T_norm, decoder=T_dec)
end

function clip_grads(grads, max_norm::Float32)
    if !(max_norm > 0f0)
        return grads, 0f0, 1f0
    end
    s = tree_sumabs2(grads)
    norm = sqrt(s)
    if !isfinite(norm)
        zeroed = tree_map(g -> (g isa AbstractArray || g isa Number) ? (g .* 0f0) : g, grads)
        return zeroed, norm, 0f0
    end
    if !(norm > max_norm)
        return grads, norm, 1f0
    end
    scale = max_norm / (norm + 1f-6)
    clipped = tree_map(g -> (g isa AbstractArray || g isa Number) ? (g .* scale) : g, grads)
    return clipped, norm, scale
end

function batch_stats(x_batch, vocab_size::Int, pad_id::Int, eos_id::Int)
    x_min = Int(minimum(x_batch))
    x_max = Int(maximum(x_batch))
    n_total = length(x_batch)
    n_pad = Int(sum(x_batch .== pad_id))
    n_eos = Int(sum(x_batch .== eos_id))
    return (; x_min, x_max, n_total, n_pad, n_eos)
end

function logsoftmax_stable(x; dims::Int)
    m = maximum(x; dims=dims)
    y = x .- m
    lse = log.(sum(exp.(y); dims=dims))
    return y .- lse
end

function _gather2d_kernel(out, a, idx, N)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= N
        @inbounds out[i] = a[idx[i], i]
    end
    return
end

function _scatter2d_kernel(da, idx, dout, N)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= N
        @inbounds da[idx[i], i] = dout[i]
    end
    return
end

function gather2d(a::AbstractMatrix, idx::AbstractVector{<:Integer})
    N = length(idx)
    out = similar(a, eltype(a), N)
    @inbounds for i in 1:N
        out[i] = a[idx[i], i]
    end
    return out
end

function gather2d(a::CUDA.CuArray, idx::CUDA.CuArray)
    N = length(idx)
    out = similar(a, eltype(a), N)
    threads = 256
    blocks = cld(N, threads)
    CUDA.@cuda threads=threads blocks=blocks _gather2d_kernel(out, a, idx, N)
    return out
end

function ChainRulesCore.rrule(::typeof(gather2d), a::AbstractMatrix, idx::AbstractVector{<:Integer})
    y = gather2d(a, idx)

    function pullback(ȳ)
        da = similar(a)
        fill!(da, zero(eltype(da)))
        @inbounds for i in 1:length(idx)
            da[idx[i], i] += ȳ[i]
        end
        return (ChainRulesCore.NoTangent(), da, ChainRulesCore.ZeroTangent())
    end

    return y, pullback
end

function ChainRulesCore.rrule(::typeof(gather2d), a::CUDA.CuArray, idx::CUDA.CuArray)
    y = gather2d(a, idx)

    function pullback(ȳ)
        da = similar(a)
        fill!(da, zero(eltype(da)))
        N = length(idx)
        threads = 256
        blocks = cld(N, threads)
        CUDA.@cuda threads=threads blocks=blocks _scatter2d_kernel(da, idx, ȳ, N)
        return (ChainRulesCore.NoTangent(), da, ChainRulesCore.ZeroTangent())
    end

    return y, pullback
end

function ChainRulesCore.rrule(::Type{CUDA.CuArray}, x::AbstractArray)
    y = CUDA.CuArray(x)
    function pullback(ȳ)
        return (ChainRulesCore.NoTangent(), ȳ)
    end
    return y, pullback
end


function load_data(data_file::AbstractString, meta_file::AbstractString)
    if !isfile(data_file)
        error("Data file $(data_file) not found!")
    end
    data_obj = JLD2.load(data_file)
    
    # Determine format
    if haskey(data_obj, "tokens") && haskey(data_obj, "reset_flags")
        # Stream Format
        data_content = data_obj["tokens"]
        reset_flags = data_obj["reset_flags"]
        is_stream = true
        is_bucketed = false
    else
        # Old Bucket Format
        data_content = data_obj["data"]
        reset_flags = Bool[] # Empty
        is_stream = false
        is_bucketed = isa(data_content, AbstractDict)
    end

    meta_src = if isfile(meta_file)
        JLD2.load(meta_file)
    else
        data_obj
    end
    params = meta_src["params"]
    char_map = get(meta_src, "char_map", Dict{Any,Any}())
    vocab = get(meta_src, "vocab", nothing)

    vocab_size = if haskey(params, "VOCAB_SIZE")
        params["VOCAB_SIZE"]
    elseif !isempty(char_map)
        length(char_map) + 3 # approx
    elseif vocab !== nothing
        length(vocab) - 1
    else
        is_stream ? maximum(data_content) : (is_bucketed ? 0 : maximum(data_content))
    end
    pad_id = params["PAD"]
    eos_id = params["EOS"]
    # Try to get MASK ID
    mask_id = get(params, "MASK", get(params, "UNK", 3))

    return data_content, reset_flags, vocab_size, is_stream, is_bucketed, pad_id, eos_id, mask_id, vocab
end

function decode_ids(ids::AbstractVector{Int}, vocab, pad_id::Int, eos_id::Int)
    vocab === nothing && return ""
    io = IOBuffer()
    for tid in ids
        if tid == eos_id
            break
        elseif tid == pad_id
            continue
        end
        idx = tid + 1
        if 1 <= idx <= length(vocab)
            write(io, string(vocab[idx]))
        else
            write(io, "<?>")
        end
    end
    return String(take!(io))
end

function print_training_samples(x::AbstractMatrix{Int}, vocab, pad_id::Int, eos_id::Int; n::Int=3)
    n_show = min(n, size(x, 2))
    for col in 1:n_show
        ids = vec(x[:, col])
        eos_pos = findfirst(==(eos_id), ids)
        pad_pos = findfirst(==(pad_id), ids)
        pad_tail = pad_pos === nothing ? false : all(==(pad_id), view(ids, pad_pos:length(ids)))
        eos_after_pad = pad_pos === nothing ? false : any(==(eos_id), view(ids, pad_pos:length(ids)))
        println("Sample $(col) | eos_pos=$(something(eos_pos, 0)) | pad_pos=$(something(pad_pos, 0)) | pad_tail=$(pad_tail) | eos_after_pad=$(eos_after_pad)")
        print(" ids: ")
        show(stdout, ids)
        println()
        s = decode_ids(ids, vocab, pad_id, eos_id)
        if !isempty(s)
            println(" text: ", s)
        end
    end
    return
end

function compute_loss(model, ps, st, x_batch, y_batch; pad_id::Int, start_id::Int, kl_weight::Float32=0f0, pred_weight::Float32=0f0)
    # Forward pass: (logits, mu, logvar, z, z_pred), st_new
    out, st_new = model(x_batch, ps, st)
    y_pred, mu, logvar, z, z_pred = out.logits, out.mu, out.logvar, out.z, out.z_pred
    
    L, B = size(x_batch)

    # 1. Reconstruction Loss (Standard)
    vocab_size = size(y_pred, 1)
    K = max(getfield(model.decoder, :block_size), 1)
    Lpad = K * cld(L, K)
    Lcap = Lpad ÷ K

    y_pred_pad = if Lpad == L
        y_pred
    else
        pad_part = similar(y_pred, vocab_size, Lpad - L, B)
        Zygote.@ignore fill!(pad_part, zero(eltype(pad_part)))
        cat(y_pred, pad_part; dims=2)
    end

    y_batch_pad = if Lpad == L
        y_batch
    else
        pad_part = similar(y_batch, Lpad - L, B)
        Zygote.@ignore fill!(pad_part, pad_id)
        cat(y_batch, pad_part; dims=1)
    end

    y_pred_flat = reshape(y_pred_pad, vocab_size, :)
    y_batch_flat = reshape(y_batch_pad, :)

    logits = eltype(y_pred_flat) <: Union{Float16,Core.BFloat16} ? Float32.(y_pred_flat) : y_pred_flat
    log_probs = logsoftmax_stable(logits; dims=1)

    mask = y_batch_flat .!= pad_id
    weights = Float32.(mask)

    picked = gather2d(log_probs, y_batch_flat)
    
    # Simple mean loss over valid tokens
    num = sum((-picked) .* weights)
    den = sum(weights)
    recon_loss = num / max(den, 1f0)

    # 2. KL Divergence Loss
    kl_per_element = -0.5f0 .* (1f0 .+ logvar .- abs2.(mu) .- exp.(logvar))
    # Sum over dim, mean over batch
    kl_per_dim = mean(sum(kl_per_element; dims=1)) # Scaler: average KL per sample
    kl_loss = kl_per_dim

    # --- Monitoring Metrics (No Gradient) ---
    var = exp.(logvar)
    mean_var = mean(var)
    
    kl_dim_vector = mean(-0.5f0 .* (1f0 .+ logvar .- abs2.(mu) .- var); dims=(2,3)) 
    # The dimensions of logvar are [Dim, N_capsules, Batch] -> dims=(2,3) averages over N and B
    
    # 3. Latent Predictive Loss (JEPA)
    # Task: z_pred[t] should predict z[t+1]
    # z: [Dim, Lcap, B]
    # Shift: z_target = z[:, 2:end, :]
    #        z_pred_in = z_pred[:, 1:end-1, :]
    # If Lcap=1, we can't do this loss (need sequence).
    pred_loss = 0f0
    if Lcap > 1
        z_target = Zygote.dropgrad(z[:, 2:end, :]) # Stop gradient on target
        z_p = z_pred[:, 1:end-1, :]
        
        diff = z_p .- z_target
        pred_loss = mean(sum(abs2, diff; dims=1))
    end

    total_loss = recon_loss + kl_weight * kl_loss + pred_weight * pred_loss

    return total_loss, st_new, (; recon=recon_loss, kl=kl_loss, pred=pred_loss, var=mean_var, kl_dim_vector=Zygote.dropgrad(kl_dim_vector))
end

function select_device()
    return CUDA.functional() ? gpu_device() : cpu_device()
end

function maybe_load_pretrained_embedding!(ps, vocab_size::Int, dim::Int, pretrain_emb_file::AbstractString)
    if isempty(pretrain_emb_file)
        return
    end

    if !isfile(pretrain_emb_file)
        error("PRETRAIN_EMB_FILE=$(pretrain_emb_file) not found")
    end

    d = JLD2.load(pretrain_emb_file)
    W = if haskey(d, "embedding")
        d["embedding"]
    elseif haskey(d, "W")
        d["W"]
    else
        error("Pretrained embedding file must contain key \"embedding\" or \"W\"")
    end

    W = Array(W)
    if size(W) == (vocab_size, dim)
        W = permutedims(W)
    end

    if size(W) != (dim, vocab_size)
        error("Embedding shape must be (dim, vocab_size) or (vocab_size, dim). Got $(size(W)). Expected dim=$dim vocab_size=$vocab_size")
    end

    T = eltype(ps.encoder.embedding.weight)
    ps.encoder.embedding.weight .= T.(W)
    return
end

# LR Scheduler: Warmup + Cosine Decay
function get_lr(step::Int, warmup_steps::Int, max_lr::Float64, total_steps::Int)
    if step <= warmup_steps
        return max_lr * (step / max(1.0, Float64(warmup_steps)))
    else
        # Avoid division by zero if total_steps <= warmup_steps
        if total_steps <= warmup_steps
            return max_lr * 0.1 # Fallback
        end
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        progress = clamp(progress, 0.0, 1.0)
        return max_lr * 0.5 * (1 + cos(π * progress))
    end
end

@noinline function get_target_lr(base_lr::Float64)
    # Default behavior: use the configured LR.
    # To adjust dynamically: replace `return base_lr` with `return 2e-4` (or any value) and save the file.
    val = base_lr
    # println("DEBUG: get_target_lr($base_lr) -> $val") # Commented out to avoid spam, uncomment to debug
    return val
end

function apply_reset!(st, reset_mask)
    function _reset_leaf(leaf_state)
        # leaf_state: Array [Hidden, ..., Batch]
        # We assume the last dimension is Batch.
        if leaf_state isa AbstractArray
            nd = ndims(leaf_state)
            if nd > 0 && size(leaf_state, nd) == length(reset_mask)
                to_reset = findall(reset_mask)
                if !isempty(to_reset)
                    # In Julia: selectdim(A, d, idx)
                    view_obj = selectdim(leaf_state, nd, to_reset)
                    fill!(view_obj, 0)
                end
            end
        end
        return leaf_state
    end
    tree_map(_reset_leaf, st)
end

# --- Stream Batch Iterator ---
# Manages B parallel streams from the 1D token array
struct StreamBatchIterator
    tokens::Vector{Int}
    resets::Vector{Bool}
    batch_size::Int
    seq_len::Int
    num_batches::Int
    offsets::Vector{Int}
    
    function StreamBatchIterator(tokens, resets, batch_size, seq_len)
        N = length(tokens)
        # We divide the total data into B segments
        segment_len = N ÷ batch_size
        offsets = [ (b-1) * segment_len + 1 for b in 1:batch_size ]
        # How many full batches can we make?
        # Each batch consumes seq_len
        # In each segment, we can fit segment_len ÷ seq_len batches
        num_batches = segment_len ÷ seq_len
        new(tokens, resets, batch_size, seq_len, num_batches, offsets)
    end
end

Base.length(it::StreamBatchIterator) = it.num_batches

function Base.iterate(it::StreamBatchIterator, state=1)
    if state > it.num_batches
        return nothing
    end
    
    B = it.batch_size
    L = it.seq_len
    
    # Prepare batch arrays
    # x_batch: [L, B]
    # reset_mask: [B] (True if a reset happened at the START of the chunk or within)
    
    curr_resets = falses(B)
    batch_ids = Matrix{Int}(undef, L, B)
    
    for b in 1:B
        start_idx = it.offsets[b] + (state - 1) * L
        end_idx = start_idx + L - 1
        
        # Copy tokens
        batch_ids[:, b] .= view(it.tokens, start_idx:end_idx)
        
        # Check for resets in this range
        if any(view(it.resets, start_idx:end_idx))
            curr_resets[b] = true
        end
    end
    
    # Return (x_batch, reset_mask)
    return ((batch_ids, curr_resets), state + 1)
end

function make_batches_impl(data_content, is_stream, is_bucketed, batch_size, rng)
    all_batches = [] # Vector{Tuple{Any, Vector{Int}}}
    if is_stream
         return []
    elseif is_bucketed
        for k in keys(data_content)
            mat = data_content[k]
            cols = size(mat, 2)
            perm = shuffle(rng, 1:cols)

            pos = 1
            while pos <= cols
                end_pos = min(pos + batch_size - 1, cols)
                batch_ids = perm[pos:end_pos]
                push!(all_batches, (k, batch_ids))
                pos = end_pos + 1
            end
        end
    else
        cols = size(data_content, 2)
        perm = shuffle(rng, 1:cols)
        pos = 1
        while pos <= cols
            end_pos = min(pos + batch_size - 1, cols)
            batch_ids = perm[pos:end_pos]
            push!(all_batches, ("default", batch_ids))
            pos = end_pos + 1
        end
    end
    return shuffle(rng, all_batches)
end

function train(cfg)
    mkpath(cfg.checkpoint_dir)

    dev = select_device()
    cpu = cpu_device()
    println("Using device: $dev")
    flush(stdout)

    data_content, reset_flags, vocab_size, is_stream, is_bucketed, pad_id, eos_id, mask_id, vocab = load_data(cfg.data_file, cfg.meta_file)

    N_total = 0
    if is_stream
        N_total = length(data_content)
        println("Loaded STREAM data. Total tokens: $N_total, Vocab: $vocab_size")
    elseif is_bucketed
        for (k, v) in data_content
            N_total += size(v, 2)
        end
        println("Loaded bucketed data. Total samples: $N_total, Vocab: $vocab_size.")
    else
        N_total = size(data_content, 2)
        println("Loaded simple data. Total samples: $N_total, Vocab: $vocab_size")
    end
    flush(stdout)

    rng = Random.default_rng()
    Random.seed!(rng, 42)

    # Initialize model
    model = HMTR_Stage1_AutoEncoder(vocab_size, cfg.dim; pad_id=pad_id, eos_id=eos_id, mamba_d_state=cfg.mamba_d_state)
    ps0, st0 = Lux.setup(rng, model)
    ps = ps0
    st = st0

    resume_opt_state = nothing
    train_step = 0
    if !isempty(cfg.resume_ckpt)
        if !isfile(cfg.resume_ckpt)
            error("resume_ckpt=$(cfg.resume_ckpt) not found")
        end
        ckpt = JLD2.load(cfg.resume_ckpt)
        ps = ckpt["ps"]
        st = ckpt["st"]
        if haskey(ckpt, "opt_state")
            resume_opt_state = ckpt["opt_state"]
        end
        if haskey(ckpt, "train_step")
            train_step = Int(ckpt["train_step"])
        end

        if !haskey(ps, :norm)
            ps = merge(ps, (norm=ps0.norm,))
        end
        if !haskey(st, :norm)
            st = merge(st, (norm=st0.norm,))
        end

        println("Resuming from ckpt=$(cfg.resume_ckpt) epoch=$(cfg.resume_epoch) step=$(cfg.resume_step) train_step=$(train_step)")
    else
        maybe_load_pretrained_embedding!(ps, vocab_size, cfg.dim, cfg.pretrain_emb_file)
    end

    ps, st, dtypes = apply_precision(ps, st, cfg)
    println("Precision | global=$(dtypes.global_dtype) encoder=$(dtypes.encoder) norm=$(dtypes.norm) decoder=$(dtypes.decoder)")

    ps = ps |> dev
    st = st |> dev

    opt = Optimisers.Adam(cfg.lr)
    opt_state = Optimisers.setup(opt, ps)
    if resume_opt_state !== nothing
        opt_state = resume_opt_state |> dev
    end

    start_id = eos_id

    # Initialize Iterator
    seq_len = cfg.seq_len
    
    local batches
    local num_batches_per_epoch
    local stream_iter

    if is_stream
        stream_iter = StreamBatchIterator(data_content, reset_flags, cfg.batch_size, seq_len)
        if cfg.max_batches > 0
            batches = Iterators.take(stream_iter, cfg.max_batches)
            num_batches_per_epoch = min(length(stream_iter), cfg.max_batches)
        else
            batches = stream_iter
            num_batches_per_epoch = length(stream_iter)
        end
    else
        batches = make_batches_impl(data_content, is_stream, is_bucketed, cfg.batch_size, rng)
        if cfg.max_batches > 0
            batches = batches[1:min(length(batches), cfg.max_batches)]
        end
        num_batches_per_epoch = length(batches)
    end

    # Scheduler Setup
    warmup_steps = cfg.warmup_steps
    total_steps = cfg.epochs * num_batches_per_epoch
    println("Training Plan: $total_steps total steps, $warmup_steps warmup steps.")
    flush(stdout)

    epoch_start = 1
    if !isempty(cfg.resume_ckpt) && cfg.resume_epoch > 0
        epoch_start = cfg.resume_epoch + 1
    end

    println("DEBUG: epoch_start=$epoch_start")
    flush(stdout)

    println("Starting training loop... (First batch may take time to compile)")
    flush(stdout)

    for epoch in epoch_start:cfg.epochs
        total_loss = 0.0
        n_updates = 0

        if !is_stream
            batches = make_batches_impl(data_content, is_stream, is_bucketed, cfg.batch_size, rng)
            if cfg.max_batches > 0
                batches = batches[1:min(length(batches), cfg.max_batches)]
            end
        end

        for (i, batch_data) in enumerate(batches)
            local x_cpu_raw, reset_mask
            
            if is_stream
                x_cpu_raw, reset_mask = batch_data
            else
                b_key, b_ids = batch_data
                 tokens_cpu = if is_bucketed
                    data_content[b_key][:, b_ids]
                else
                    data_content[:, b_ids]
                end
                
                if size(tokens_cpu, 2) == 0
                    continue
                end
                
                x_cpu_raw = Matrix{Int}(tokens_cpu)
                reset_mask = falses(size(x_cpu_raw, 2)) # No resets for bucketed
            end

            # Update LR
            target_lr = get_target_lr(cfg.lr)
            current_lr = get_lr(train_step + 1, warmup_steps, target_lr, total_steps)
            Optimisers.adjust!(opt_state, current_lr)

            # --- State Reset Logic ---
            apply_reset!(st, reset_mask)
            
            x_batch = x_cpu_raw |> dev
            y_batch = x_batch # AutoEncoder target

            # Gradient
            (loss, st_new, internals), back = Zygote.pullback(
                p -> compute_loss(model, p, st, x_batch, y_batch; pad_id=pad_id, start_id=start_id, kl_weight=Float32(cfg.kl_weight), pred_weight=Float32(cfg.pred_weight)), ps
            )

            loss_val = Float32(loss)
            
            # Check for loss spike / NaN
            spike = !(isfinite(loss_val)) || (loss_val > Float32(cfg.loss_spike_threshold))
            if spike
                stats = batch_stats(x_batch, vocab_size, pad_id, eos_id)
                pad_frac = stats.n_pad / max(stats.n_total, 1)
                eos_frac = stats.n_eos / max(stats.n_total, 1)
                @printf "SPIKE Epoch %d Step %d Loss %.4f | x[min=%d max=%d] pad=%.3f eos=%.3f\n" epoch i loss_val stats.x_min stats.x_max pad_frac eos_frac
                if stats.x_min < 1 || stats.x_max > vocab_size
                    @printf "SPIKE TokenId out of range: expected [1,%d]\n" vocab_size
                end
                if cfg.skip_on_spike
                    continue
                end
            end

            grads = back((one(loss), nothing, nothing))[1]
            grads, grad_norm, grad_scale = clip_grads(grads, Float32(cfg.grad_clip_norm))
            if grad_scale < 1f0 && (i % 50 == 0 || grad_scale < 0.2f0)
                @printf "GradClip Epoch %d Step %d | norm=%.4f scale=%.6f\n" epoch i grad_norm grad_scale
            end

            opt_state, ps = Optimisers.update(opt_state, ps, grads)
            st = st_new
            total_loss += loss_val
            n_updates += 1
            train_step += 1

            if i % 50 == 0
                au = count(internals.kl_dim_vector .> 0.01f0)
                @printf "Epoch %d [%d/%d] Loss: %.4f (Recon: %.4f | KL: %.4f | Pred: %.4f) | Var: %.4f | AU: %d | LR: %.2e\n" epoch i num_batches_per_epoch loss_val internals.recon internals.kl internals.pred internals.var au current_lr
            end

            if cfg.save_every > 0 && (i % cfg.save_every == 0)
                jldsave(
                    joinpath(cfg.checkpoint_dir, "$(cfg.checkpoint_prefix)_epoch$(epoch)_step$(i).jld2");
                    ps=ps |> cpu,
                    st=st |> cpu,
                    opt_state=opt_state |> cpu,
                    epoch=epoch,
                    step=i,
                    train_step=train_step,
                    mamba_d_state=cfg.mamba_d_state,
                )
            end
        end

        avg_loss = total_loss / max(n_updates, 1)
        println("Epoch $epoch Completed. Avg Loss: $avg_loss")
        flush(stdout)

        # Save Checkpoint
        jldsave(
            joinpath(cfg.checkpoint_dir, "$(cfg.checkpoint_prefix)_epoch$epoch.jld2");
            ps=ps |> cpu,
            st=st |> cpu,
            opt_state=opt_state |> cpu,
            epoch=epoch,
            step=0,
            train_step=train_step,
            mamba_d_state=cfg.mamba_d_state,
        )
    end
end


function inspect_data(cfg)
    data_content, reset_flags, vocab_size, is_stream, is_bucketed, pad_id, eos_id, mask_id, vocab = load_data(cfg.data_file, cfg.meta_file)

    N_total = 0
    if is_bucketed
        for (k, v) in data_content
            N_total += size(v, 2)
        end
        println("Inspect Bucketed Data. Total samples: $N_total, Vocab: $vocab_size")

        # Show a few from each bucket
        for k in keys(data_content)
            println("\n--- Bucket $k ---")
            mat = data_content[k]
            cols = size(mat, 2)
            n_show = min(cfg.inspect_n, cols)
            rng = Random.default_rng()
            Random.seed!(rng, cfg.inspect_seed)
            cols_idx = shuffle(rng, 1:cols)[1:n_show]
            x = Matrix{Int}(mat[:, cols_idx])
            print_training_samples(x, vocab, pad_id, eos_id; n=n_show)
        end
    else
        N_total = size(data_content, 2)
        n = min(cfg.inspect_n, N_total)
        println("Inspect Data. Total samples: $N_total")
        rng = Random.default_rng()
        Random.seed!(rng, cfg.inspect_seed)
        cols = shuffle(rng, 1:N_total)[1:n]
        x = Matrix{Int}(data_content[:, cols])
        print_training_samples(x, vocab, pad_id, eos_id; n=n)
    end
    return
end

function train_stage1(args::Vector{String})
    if "--help" in args || "-h" in args
        println("Usage: train_stage1 [options]")
        println("Options:")
        println("  --data-file <path>        Path to processed data file (default: \$DATA_FILE)")
        println("  --meta-file <path>        Path to metadata file (default: derived from data-file)")
        println("  --checkpoint-dir <path>   Directory to save checkpoints")
        println("  --epochs <int>            Number of epochs")
        println("  --batch-size <int>        Batch size")
        println("  --lr <float>              Learning rate")
        println("  --dim <int>               Model dimension")
        println("  --mamba-d-state <int>     Mamba d_state (default: $MAMBA_D_STATE)")
        println("  --warmup-steps <int>      Warmup steps (default: $WARMUP_STEPS)")
        println("  --seq-len <int>           Stream chunk length (default: $SEQ_LEN)")
        println("  --inspect-data            Inspect data instead of training")
        return
    end
    cli = Utils.parse_cli_args(args)
    cfg = resolve_config(cli)
    if cfg.inspect_data
        inspect_data(cfg)
    else
        train(cfg)
    end
end

end # module Train
