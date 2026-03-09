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
const KL_WEIGHT = parse(Float64, get(ENV, "KL_WEIGHT", "0.0"))
const PRED_WEIGHT = parse(Float64, get(ENV, "PRED_WEIGHT", "1.0"))
const VAR_DIR_WEIGHT = parse(Float64, get(ENV, "VAR_DIR_WEIGHT", "0.15"))
const VAR_MAG_WEIGHT = parse(Float64, get(ENV, "VAR_MAG_WEIGHT", "0.01"))
const VAR_MAG_LOW = parse(Float64, get(ENV, "VAR_MAG_LOW", "0.2"))
const VAR_MAG_HIGH = parse(Float64, get(ENV, "VAR_MAG_HIGH", "3.5"))
const AUTO_LOSS_BALANCE = parse(Int, get(ENV, "AUTO_LOSS_BALANCE", "1"))
const TARGET_PRED_RATIO = parse(Float64, get(ENV, "TARGET_PRED_RATIO", "0.25"))
const TARGET_VAR_DIR_RATIO = parse(Float64, get(ENV, "TARGET_VAR_DIR_RATIO", "0.10"))
const PRED_WARMUP_FRAC = parse(Float64, get(ENV, "PRED_WARMUP_FRAC", "0.10"))
const VAR_DIR_START_FRAC = parse(Float64, get(ENV, "VAR_DIR_START_FRAC", "0.10"))
const VAR_DIR_FULL_FRAC = parse(Float64, get(ENV, "VAR_DIR_FULL_FRAC", "0.40"))
const PRED_DECAY_START_FRAC = parse(Float64, get(ENV, "PRED_DECAY_START_FRAC", "0.75"))
const PRED_DECAY_END_SCALE = parse(Float64, get(ENV, "PRED_DECAY_END_SCALE", "0.60"))
const TEACHER_FORCING = parse(Int, get(ENV, "TEACHER_FORCING", "0"))

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
const RESUME_META_FILE = get(ENV, "RESUME_META_FILE", "")
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
    # 1. Resolve resume_ckpt first to load config defaults
    resume_ckpt = string(get(cli, :resume_ckpt, get(cli, :resume, get(cli, :resume_from, RESUME_CKPT))))
    if resume_ckpt == "true"
        resume_ckpt = ""
    end
    
    checkpoint_dir = get(cli, :checkpoint_dir, CHECKPOINT_DIR)
    # Smart path resolution for resume_ckpt
    if !isempty(resume_ckpt) && !isfile(resume_ckpt)
        alt_path = joinpath(checkpoint_dir, basename(resume_ckpt))
        if isfile(alt_path)
            resume_ckpt = alt_path
        elseif isfile(joinpath(checkpoint_dir, resume_ckpt))
            resume_ckpt = joinpath(checkpoint_dir, resume_ckpt)
        end
    end

    # 2. Load config from checkpoint if available
    ckpt_meta_data = nothing
    if !isempty(resume_ckpt) && isfile(resume_ckpt)
        try
            JLD2.jldopen(resume_ckpt, "r") do file
                if haskey(file, "config")
                    ckpt_config = file["config"]
                    # Merge ckpt_config into cli (CLI takes precedence)
                    # We only add keys that are NOT in cli
                    if ckpt_config isa NamedTuple || ckpt_config isa AbstractDict
                        for k in keys(ckpt_config)
                            if !haskey(cli, k)
                                cli[k] = ckpt_config[k]
                            end
                        end
                    end
                end
                if haskey(file, "meta_data")
                    ckpt_meta_data = file["meta_data"]
                end
            end
            println("Loaded configuration from checkpoint: $resume_ckpt")
        catch e
            @warn "Failed to load config from checkpoint: $e"
        end
    end

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
    var_dir_weight = parse(Float64, string(get(cli, :var_dir_weight, VAR_DIR_WEIGHT)))
    var_mag_weight = parse(Float64, string(get(cli, :var_mag_weight, VAR_MAG_WEIGHT)))
    var_mag_low = parse(Float64, string(get(cli, :var_mag_low, VAR_MAG_LOW)))
    var_mag_high = parse(Float64, string(get(cli, :var_mag_high, VAR_MAG_HIGH)))
    auto_loss_balance = parse_bool(:auto_loss_balance, AUTO_LOSS_BALANCE)
    target_pred_ratio = parse(Float64, string(get(cli, :target_pred_ratio, TARGET_PRED_RATIO)))
    target_var_dir_ratio = parse(Float64, string(get(cli, :target_var_dir_ratio, TARGET_VAR_DIR_RATIO)))
    pred_warmup_frac = parse(Float64, string(get(cli, :pred_warmup_frac, PRED_WARMUP_FRAC)))
    var_dir_start_frac = parse(Float64, string(get(cli, :var_dir_start_frac, VAR_DIR_START_FRAC)))
    var_dir_full_frac = parse(Float64, string(get(cli, :var_dir_full_frac, VAR_DIR_FULL_FRAC)))
    pred_decay_start_frac = parse(Float64, string(get(cli, :pred_decay_start_frac, PRED_DECAY_START_FRAC)))
    pred_decay_end_scale = parse(Float64, string(get(cli, :pred_decay_end_scale, PRED_DECAY_END_SCALE)))
    pred_warmup_frac = clamp(pred_warmup_frac, 0.0, 1.0)
    var_dir_start_frac = clamp(var_dir_start_frac, 0.0, 1.0)
    var_dir_full_frac = clamp(var_dir_full_frac, var_dir_start_frac, 1.0)
    pred_decay_start_frac = clamp(pred_decay_start_frac, 0.0, 1.0)
    pred_decay_end_scale = clamp(pred_decay_end_scale, 0.0, 1.0)
    var_mag_low = max(var_mag_low, 0.0)
    var_mag_high = max(var_mag_high, var_mag_low + 1e-6)
    teacher_forcing = parse_bool(:teacher_forcing, TEACHER_FORCING)
    inspect_seed = parse(Int, string(get(cli, :inspect_seed, INSPECT_SEED)))
    dtype = string(get(cli, :dtype, DTYPE))
    encoder_dtype = string(get(cli, :encoder_dtype, ENCODER_DTYPE))
    norm_dtype = string(get(cli, :norm_dtype, NORM_DTYPE))
    decoder_dtype = string(get(cli, :decoder_dtype, DECODER_DTYPE))
    seq_len = parse(Int, string(get(cli, :seq_len, SEQ_LEN)))

    resume_meta_file = string(get(cli, :resume_meta_file, get(cli, :resume_meta, RESUME_META_FILE)))

    if isempty(resume_meta_file) && !isempty(resume_ckpt)
        resume_meta_file = replace(resume_ckpt, r"\.jld2$" => "_meta.jld2")
    end
    if !isempty(resume_meta_file) && !isfile(resume_meta_file)
        alt_meta_path = joinpath(checkpoint_dir, basename(resume_meta_file))
        if isfile(alt_meta_path)
            resume_meta_file = alt_meta_path
        else
            resume_meta_file = ""
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
        resume_meta_file,
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
        var_dir_weight,
        var_mag_weight,
        var_mag_low,
        var_mag_high,
        auto_loss_balance,
        target_pred_ratio,
        target_var_dir_ratio,
        pred_warmup_frac,
        var_dir_start_frac,
        var_dir_full_frac,
        pred_decay_start_frac,
        pred_decay_end_scale,
        teacher_forcing,
        seq_len,
        meta_data = ckpt_meta_data,
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


function load_data(data_file::AbstractString, meta_file::AbstractString; meta_data_override=nothing)
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

    meta_src = if meta_data_override !== nothing
        meta_data_override
    elseif isfile(meta_file)
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

    return data_content, reset_flags, vocab_size, is_stream, is_bucketed, pad_id, eos_id, mask_id, vocab, meta_src
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

function _phase_ramp(progress::Float32, start::Float32, full::Float32)
    if progress <= start
        return 0f0
    elseif progress >= full
        return 1f0
    else
        return (progress - start) / max(full - start, 1f-6)
    end
end

function _phase_decay(progress::Float32, start::Float32, end_scale::Float32)
    if progress <= start
        return 1f0
    else
        ratio = (progress - start) / max(1f0 - start, 1f-6)
        ratio = clamp(ratio, 0f0, 1f0)
        return 1f0 - (1f0 - end_scale) * ratio
    end
end

function compute_adaptive_loss_weights(cfg, train_step::Int, total_steps::Int, ema)
    progress = Float32(train_step) / max(Float32(total_steps), 1f0)
    pred_phase = _phase_ramp(progress, 0f0, Float32(cfg.pred_warmup_frac))
    pred_decay = _phase_decay(progress, Float32(cfg.pred_decay_start_frac), Float32(cfg.pred_decay_end_scale))
    var_dir_phase = _phase_ramp(progress, Float32(cfg.var_dir_start_frac), Float32(cfg.var_dir_full_frac))

    pred_weight = Float32(cfg.pred_weight) * pred_phase * pred_decay
    var_dir_weight = Float32(cfg.var_dir_weight) * var_dir_phase
    var_mag_weight = Float32(cfg.var_mag_weight)

    if cfg.auto_loss_balance
        recon_ema = max(ema.recon, 1f-6)
        pred_ratio = ema.pred / recon_ema
        var_dir_ratio = ema.var_dir / recon_ema
        pred_gain = clamp(Float32(cfg.target_pred_ratio) / max(pred_ratio, 1f-4), 0.5f0, 2f0)
        var_dir_gain = clamp(Float32(cfg.target_var_dir_ratio) / max(var_dir_ratio, 1f-4), 0.5f0, 2f0)
        pred_weight *= pred_gain
        var_dir_weight *= var_dir_gain

        abs_logvar = ema.abs_logvar
        if abs_logvar > Float32(cfg.var_mag_high)
            var_mag_weight *= clamp(abs_logvar / Float32(cfg.var_mag_high), 1f0, 4f0)
        elseif abs_logvar < Float32(cfg.var_mag_low)
            var_mag_weight *= clamp(Float32(cfg.var_mag_low) / max(abs_logvar, 1f-4), 1f0, 4f0)
        end
    end

    return (; pred=pred_weight, var_dir=var_dir_weight, var_mag=var_mag_weight)
end

function update_loss_ema(ema, internals; alpha::Float32=0.05f0)
    one_minus = 1f0 - alpha
    return (
        recon=one_minus * ema.recon + alpha * Float32(internals.recon),
        pred=one_minus * ema.pred + alpha * Float32(internals.pred),
        var_dir=one_minus * ema.var_dir + alpha * Float32(internals.var_dir),
        var_mag=one_minus * ema.var_mag + alpha * Float32(internals.var_mag),
        abs_logvar=one_minus * ema.abs_logvar + alpha * Float32(internals.abs_logvar),
    )
end

function compute_loss(model, ps, st, x_batch, y_batch; pad_id::Int, eos_id::Int, kl_weight::Float32=0f0, pred_weight::Float32=0f0, var_dir_weight::Float32=0f0, var_mag_weight::Float32=0f0, var_mag_low::Float32=0.2f0, var_mag_high::Float32=3.5f0, teacher_forcing::Bool=false)
    out, st_new = model(x_batch, ps, st; teacher_forcing=teacher_forcing)
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

    kl_loss = zero(recon_loss)
    pred_loss = zero(recon_loss)
    var_dir_loss = zero(recon_loss)

    kl_per_element = -0.5f0 .* (1f0 .+ logvar .- abs2.(mu) .- exp.(logvar))
    kl_per_dim = mean(sum(kl_per_element; dims=1))
    kl_loss = kl_per_dim
    var = exp.(logvar)
    mean_var = mean(var)
    
    kl_dim_vector = mean(-0.5f0 .* (1f0 .+ logvar .- abs2.(mu) .- var); dims=(2,3)) 
    if Lcap > 1
        z_target = Zygote.dropgrad(z[:, 2:end, :])
        z_p = z_pred[:, 1:end-1, :]
        
        diff = z_p .- z_target
        pred_loss = mean(mean(abs2, diff; dims=1))

        eos_mask_tokens = (y_batch_pad .== eos_id)
        eos_mask_3d = reshape(eos_mask_tokens, K, Lcap, B)
        capsule_has_eos = dropdims(any(eos_mask_3d; dims=1); dims=1)
        valid_transition = .!(capsule_has_eos[1:end-1, :])

        logvar_src = logvar[:, 1:end-1, :]
        logvar_tgt = Zygote.dropgrad(logvar[:, 2:end, :])
        src_normsq = sum(abs2, logvar_src; dims=1) .+ 1f-6
        tgt_normsq = sum(abs2, logvar_tgt; dims=1) .+ 1f-6
        dot_product = sum(logvar_src .* logvar_tgt; dims=1)
        cos_sim = dot_product ./ sqrt.(src_normsq .* tgt_normsq)
        per_transition_loss = 1f0 .- cos_sim
        masked_loss = per_transition_loss .* reshape(Float32.(valid_transition), 1, Lcap - 1, B)
        num_valid = sum(valid_transition)
        var_dir_loss = sum(masked_loss) / max(Float32(num_valid), 1f0)
    end

    abs_logvar = abs.(logvar)
    high_penalty = NNlib.relu.(abs_logvar .- var_mag_high)
    low_penalty = NNlib.relu.(var_mag_low .- abs_logvar)
    var_mag_loss = mean(abs2, high_penalty) + mean(abs2, low_penalty)

    total_loss = recon_loss + kl_weight * kl_loss + pred_weight * pred_loss + var_dir_weight * var_dir_loss + var_mag_weight * var_mag_loss
    
    z_mean_val = mean(mu)
    z_std_val = mean(exp.(0.5f0 .* logvar))
    abs_logvar_mean = mean(abs_logvar)

    return total_loss, st_new, (; recon=recon_loss, kl=kl_loss, pred=pred_loss, var_dir=var_dir_loss, var_mag=var_mag_loss, var=mean_var, z_mean=z_mean_val, z_std=z_std_val, abs_logvar=abs_logvar_mean, kl_dim=Zygote.dropgrad(kl_dim_vector))
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
mutable struct StreamBatchIterator
    tokens::Vector{Int}
    resets::Vector{Bool}
    batch_size::Int
    seq_len::Int
    num_batches::Int
    offsets::Vector{Int}
    segment_ends::Vector{Int}
    positions::Vector{Int}
    block_size::Int
    eos_id::Int
    pad_id::Int
    
    function StreamBatchIterator(tokens, resets, batch_size, seq_len, block_size, eos_id, pad_id)
        N = length(tokens)
        segment_len = N ÷ batch_size
        offsets = [ (b-1) * segment_len + 1 for b in 1:batch_size ]
        segment_ends = [offsets[b] + segment_len - 1 for b in 1:batch_size]
        positions = copy(offsets)

        expanded_lengths = Vector{Int}(undef, batch_size)
        for b in 1:batch_size
            expanded = 0
            for idx in offsets[b]:segment_ends[b]
                expanded += 1
                if tokens[idx] == eos_id
                    rem = block_size - (expanded % block_size)
                    expanded += rem == block_size ? 0 : rem
                end
            end
            expanded_lengths[b] = expanded
        end
        num_batches = minimum(expanded_lengths) ÷ seq_len
        new(tokens, resets, batch_size, seq_len, num_batches, offsets, segment_ends, positions, block_size, eos_id, pad_id)
    end
end

Base.length(it::StreamBatchIterator) = it.num_batches

function Base.iterate(it::StreamBatchIterator, state=1)
    if state == 1
        it.positions .= it.offsets
    end
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
    fill!(batch_ids, it.pad_id)
    
    for b in 1:B
        out_pos = 1
        while out_pos <= L
            src_pos = it.positions[b]
            if src_pos > it.segment_ends[b]
                break
            end
            tok = it.tokens[src_pos]
            rst = it.resets[src_pos]
            it.positions[b] = src_pos + 1
            batch_ids[out_pos, b] = tok
            curr_resets[b] = curr_resets[b] || rst
            if tok == it.eos_id
                out_pos = ((out_pos - 1) ÷ it.block_size + 1) * it.block_size + 1
            else
                out_pos += 1
            end
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

    data_content, reset_flags, vocab_size, is_stream, is_bucketed, pad_id, eos_id, mask_id, vocab, meta_data = load_data(cfg.data_file, cfg.meta_file; meta_data_override=cfg.meta_data)

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
        ps_ckpt = ckpt["ps"]
        st = ckpt["st"]
        if haskey(ckpt, "opt_state")
            resume_opt_state = ckpt["opt_state"]
        end
        if haskey(ckpt, "train_step")
            train_step = Int(ckpt["train_step"])
        end

        vocab_resized = false
        vocab_ckpt = size(ps_ckpt.encoder.embedding.weight, 2)
        if vocab_ckpt != vocab_size
            println("Warning: Vocab size mismatch! Checkpoint=$(vocab_ckpt), New=$(vocab_size). Performing char-aligned merge.")
            vocab_resized = true
            
            # Try to load char_maps for character-level alignment
            # We need: old_char_map (from checkpoint) and new_char_map (from current data)
            old_char_map = nothing
            new_char_map = nothing
            
            # Load old char_map from a sibling _meta.jld2 if available  
            ckpt_meta_path = replace(cfg.resume_ckpt, r"\.jld2$" => "_meta.jld2")
            old_meta_path = !isempty(cfg.resume_meta_file) ? cfg.resume_meta_file : ckpt_meta_path
            if isfile(old_meta_path)
                old_meta = JLD2.load(old_meta_path)
                if haskey(old_meta, "char_map")
                    old_char_map = Dict{Char,Int}(old_meta["char_map"])
                    println("Loaded old char_map from: $old_meta_path ($(length(old_char_map)) chars)")
                end
            end
            
            # Load new char_map from new data meta_file
            if isfile(cfg.meta_file)
                new_meta = JLD2.load(cfg.meta_file)
                if haskey(new_meta, "char_map")
                    new_char_map = Dict{Char,Int}(new_meta["char_map"])
                    println("Loaded new char_map from: $(cfg.meta_file) ($(length(new_char_map)) chars)")
                end
            end
            
            # Determine how to build the weight transfer function
            if old_char_map !== nothing && new_char_map !== nothing
                println("Performing character-aligned embedding merge (union of both vocabs).")

                # Helper: copy embeddings by char identity
                function merge_embed_by_char(w_old::AbstractMatrix, w_new_shape::AbstractMatrix, old_cm, new_cm)
                    w_new = copy(w_new_shape)  # start from random init
                    for (c, old_id) in old_cm
                        new_id = get(new_cm, c, nothing)
                        if new_id !== nothing
                            old_col = old_id + 1
                            new_col = new_id + 1
                            if old_col >= 1 && old_col <= size(w_old, 2) && new_col >= 1 && new_col <= size(w_new, 2)
                                w_new[:, new_col] .= w_old[:, old_col]
                            end
                        end
                    end
                    return w_new
                end
                function merge_proj_by_char(w_old::AbstractMatrix, b_old::AbstractArray, w_new_shape, b_new_shape, old_cm, new_cm)
                    w_new = copy(w_new_shape)
                    b_new = copy(b_new_shape)
                    for (c, old_id) in old_cm
                        new_id = get(new_cm, c, nothing)
                        if new_id !== nothing
                            old_row = old_id + 1
                            new_row = new_id + 1
                            if old_row >= 1 && old_row <= size(w_old, 1) && new_row >= 1 && new_row <= size(w_new, 1)
                                w_new[new_row, :] .= w_old[old_row, :]
                                if ndims(b_old) >= 2
                                    b_new[new_row, 1] = b_old[old_row, 1]
                                else
                                    b_new[new_row] = b_old[old_row]
                                end
                            end
                        end
                    end
                    return w_new, b_new
                end

                new_enc_embed = merge_embed_by_char(
                    ps_ckpt.encoder.embedding.weight, ps0.encoder.embedding.weight,
                    old_char_map, new_char_map)
                ps_enc_new = merge(ps_ckpt.encoder, (embedding=(weight=new_enc_embed,),))
                ps_new = merge(ps_ckpt, (encoder=ps_enc_new,))
                
                if haskey(ps_ckpt, :decoder)
                    new_dec_embed = merge_embed_by_char(
                        ps_ckpt.decoder.embedding.weight, ps0.decoder.embedding.weight,
                        old_char_map, new_char_map)
                    new_dec_proj_w, new_dec_proj_b = merge_proj_by_char(
                        ps_ckpt.decoder.proj.weight, ps_ckpt.decoder.proj.bias,
                        ps0.decoder.proj.weight, ps0.decoder.proj.bias,
                        old_char_map, new_char_map)
                    ps_dec_new = merge(ps_ckpt.decoder, (embedding=(weight=new_dec_embed,), proj=(weight=new_dec_proj_w, bias=new_dec_proj_b)))
                    ps = merge(ps_new, (decoder=ps_dec_new,))
                else
                    ps = ps_new
                end
            else
                # Fallback: positional copy (old behavior, less accurate if vocab order changed)
                println("char_map not available for alignment; falling back to positional copy.")
                n_copy = min(vocab_ckpt, vocab_size)
                
                new_enc_embed = copy(ps0.encoder.embedding.weight)
                new_enc_embed[:, 1:n_copy] .= ps_ckpt.encoder.embedding.weight[:, 1:n_copy]
                ps_enc_new = merge(ps_ckpt.encoder, (embedding=(weight=new_enc_embed,),))
                ps_new = merge(ps_ckpt, (encoder=ps_enc_new,))
                
                if haskey(ps_ckpt, :decoder)
                    new_dec_embed = copy(ps0.decoder.embedding.weight)
                    new_dec_embed[:, 1:n_copy] .= ps_ckpt.decoder.embedding.weight[:, 1:n_copy]
                    new_dec_proj_w = copy(ps0.decoder.proj.weight)
                    new_dec_proj_b = copy(ps0.decoder.proj.bias)
                    if haskey(ps_ckpt.decoder, :proj)
                        new_dec_proj_w[1:n_copy, :] .= ps_ckpt.decoder.proj.weight[1:n_copy, :]
                        if length(size(ps_ckpt.decoder.proj.bias)) >= 2
                            new_dec_proj_b[1:n_copy, 1] .= ps_ckpt.decoder.proj.bias[1:n_copy, 1]
                        else
                            new_dec_proj_b[1:n_copy] .= ps_ckpt.decoder.proj.bias[1:n_copy]
                        end
                    end
                    ps_dec_new = merge(ps_ckpt.decoder, (embedding=(weight=new_dec_embed,), proj=(weight=new_dec_proj_w, bias=new_dec_proj_b)))
                    ps = merge(ps_new, (decoder=ps_dec_new,))
                else
                    ps = ps_new
                end
            end
        else
            ps = ps_ckpt
        end

        if vocab_resized && resume_opt_state !== nothing
            println("Opt state reset due to vocab resize")
            resume_opt_state = nothing
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

    loss_ema = (recon=1f0, pred=1f0, var_dir=1f0, var_mag=1f0, abs_logvar=1f0)

    # Initialize Iterator
    seq_len = cfg.seq_len
    block_size = max(getfield(model.encoder, :block_size), 1)
    
    local batches
    local num_batches_per_epoch
    local stream_iter

    if is_stream
        stream_iter = StreamBatchIterator(data_content, reset_flags, cfg.batch_size, seq_len, block_size, eos_id, pad_id)
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

        if is_stream
            stream_iter = StreamBatchIterator(data_content, reset_flags, cfg.batch_size, seq_len, block_size, eos_id, pad_id)
            if cfg.max_batches > 0
                batches = Iterators.take(stream_iter, cfg.max_batches)
            else
                batches = stream_iter
            end
        else
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
                reset_mask = trues(size(x_cpu_raw, 2))
            end

            # Update LR
            target_lr = get_target_lr(cfg.lr)
            current_lr = get_lr(train_step + 1, warmup_steps, target_lr, total_steps)
            Optimisers.adjust!(opt_state, current_lr)
            loss_weights = compute_adaptive_loss_weights(cfg, train_step, total_steps, loss_ema)

            # --- State Reset Logic ---
            apply_reset!(st, reset_mask)
            
            x_batch = x_cpu_raw |> dev
            y_batch = x_batch # AutoEncoder target

            # Gradient
            st_holder = Ref{Any}(nothing)
            internals_holder = Ref{Any}(nothing)
            loss, grad_tuple = Zygote.withgradient(ps) do p
                l, st_tmp, internals_tmp = compute_loss(
                    model, p, st, x_batch, y_batch;
                    pad_id=pad_id,
                    eos_id=eos_id,
                    kl_weight=Float32(cfg.kl_weight),
                    pred_weight=loss_weights.pred,
                    var_dir_weight=loss_weights.var_dir,
                    var_mag_weight=loss_weights.var_mag,
                    var_mag_low=Float32(cfg.var_mag_low),
                    var_mag_high=Float32(cfg.var_mag_high),
                    teacher_forcing=cfg.teacher_forcing,
                )
                st_holder[] = st_tmp
                internals_holder[] = internals_tmp
                return l
            end
            st_new = st_holder[]
            internals = internals_holder[]
            grads = grad_tuple[1]

            loss_val = Float32(loss)
            
            # Check for loss spike / NaN
            spike = !isfinite(loss_val) || loss_val > Float32(cfg.loss_spike_threshold)
            if spike
                stats = batch_stats(x_batch, vocab_size, pad_id, eos_id)
                pad_frac = stats.n_pad / max(stats.n_total, 1)
                eos_frac = stats.n_eos / max(stats.n_total, 1)
                @printf "SPIKE Epoch %d Step %d Loss %.4f | x[min=%d max=%d] pad=%.3f eos=%.3f\n" epoch i loss_val stats.x_min stats.x_max pad_frac eos_frac
                if stats.x_min < 1 || stats.x_max > vocab_size
                    @printf "SPIKE TokenId out of range: expected [1,%d]\n" vocab_size
                end
                
                # If NaN, we MUST reset the state, otherwise future steps will also be NaN
                if !isfinite(loss_val)
                    println("WARNING: Loss is NaN/Inf. Resetting model state to clear corrupted hidden states.")
                    st = st0 |> dev
                end

                if cfg.skip_on_spike
                    continue
                end
            end

            grads, grad_norm, grad_scale = clip_grads(grads, Float32(cfg.grad_clip_norm))
            if grad_scale < 1f0 && (i % 50 == 0 || grad_scale < 0.2f0)
                @printf "GradClip Epoch %d Step %d | norm=%.4f scale=%.6f\n" epoch i grad_norm grad_scale
            end

            opt_state, ps = Optimisers.update(opt_state, ps, grads)
            st = st_new
            total_loss += loss_val
            n_updates += 1
            train_step += 1
            loss_ema = update_loss_ema(loss_ema, internals)

            if i % 50 == 0
                au = count(internals.kl_dim .> 0.01f0)
                @printf "Epoch %d [%d/%d] Loss: %.4f (Recon: %.4f | Pred: %.4f | VarDir: %.4f | VarMag: %.4f | KL: %.4f) | w[p=%.3f vd=%.3f vm=%.3f] | |logvar|: %.4f | Var: %.4f | z_μ: %.4f | z_σ: %.4f | AU: %d | |g|: %.4f | LR: %.2e\n" epoch i num_batches_per_epoch loss_val internals.recon internals.pred internals.var_dir internals.var_mag internals.kl loss_weights.pred loss_weights.var_dir loss_weights.var_mag internals.abs_logvar internals.var internals.z_mean internals.z_std au grad_norm current_lr
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
                    config=cfg,
                    meta_data=meta_data,
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
            config=cfg,
            meta_data=meta_data,
        )
    end
end


function inspect_data(cfg)
    data_content, reset_flags, vocab_size, is_stream, is_bucketed, pad_id, eos_id, mask_id, vocab, _ = load_data(cfg.data_file, cfg.meta_file)

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
        println("  --resume-meta-file <path> Path to old meta for vocab alignment")
        println("  --epochs <int>            Number of epochs")
        println("  --batch-size <int>        Batch size")
        println("  --lr <float>              Learning rate")
        println("  --dim <int>               Model dimension")
        println("  --mamba-d-state <int>     Mamba d_state (default: $MAMBA_D_STATE)")
        println("  --warmup-steps <int>      Warmup steps (default: $WARMUP_STEPS)")
        println("  --pred-weight <float>     Base prediction loss weight")
        println("  --var-dir-weight <float>  Base variance direction loss weight")
        println("  --var-mag-weight <float>  Base variance magnitude loss weight")
        println("  --auto-loss-balance <0|1> Enable adaptive loss balancing")
        println("  --seq-len <int>           Stream chunk length (default: $SEQ_LEN)")
        println("  --teacher-forcing         Enable teacher forcing")
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
