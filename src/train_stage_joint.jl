module TrainStageJoint

using Lux
using Random
using JLD2
using Optimisers
using Zygote
using Printf
using NNlib
using CUDA
using LuxCUDA
using Dates
using Statistics
using ..Utils
using ..Model
import ..TrainStage1
import ..PrecisionRuntime

export train_stage_joint

const PROJECT_ROOT = normpath(joinpath(@__DIR__, ".."))
const DATA_DIR = get(ENV, "DATA_DIR", joinpath(PROJECT_ROOT, "data"))
const DATA_FILE = get(ENV, "DATA_FILE", joinpath(DATA_DIR, "processed.jld2"))
const META_FILE = get(ENV, "META_FILE", "")
const CHECKPOINT_DIR = get(ENV, "CHECKPOINT_DIR", joinpath(PROJECT_ROOT, "checkpoints"))
const CHECKPOINT_PREFIX = get(ENV, "CHECKPOINT_PREFIX", "ckpt_stage_joint")
const EPOCHS = parse(Int, get(ENV, "EPOCHS", "5"))
const BATCH_SIZE = parse(Int, get(ENV, "BATCH_SIZE", "24"))
const LR = parse(Float64, get(ENV, "LR", "1e-3"))
const MAX_BATCHES = parse(Int, get(ENV, "MAX_BATCHES", "0"))
const SAVE_EVERY = parse(Int, get(ENV, "SAVE_EVERY", "0"))
const ADD_TIMESTAMP = parse(Int, get(ENV, "ADD_TIMESTAMP", "1"))
const GRAD_CLIP_NORM = parse(Float64, get(ENV, "GRAD_CLIP_NORM", "5.0"))
const LOSS_SPIKE_THRESHOLD = parse(Float64, get(ENV, "LOSS_SPIKE_THRESHOLD", "10.0"))
const SKIP_ON_SPIKE = parse(Int, get(ENV, "SKIP_ON_SPIKE", "1"))
const WARMUP_STEPS = parse(Int, get(ENV, "WARMUP_STEPS", "500"))
const MODEL_DIM = parse(Int, get(ENV, "MODEL_DIM", "256"))
const MAMBA_D_STATE = parse(Int, get(ENV, "MAMBA_D_STATE", "32"))
const BLOCK_SIZE = parse(Int, get(ENV, "BLOCK_SIZE", "8"))
const SEQ_LEN = parse(Int, get(ENV, "SEQ_LEN", "512"))
const FORCE_CPU = parse(Int, get(ENV, "FORCE_CPU", "0"))

const K_STREAMS = parse(Int, get(ENV, "K_STREAMS", "4"))
const HEADS = parse(Int, get(ENV, "HEADS", "8"))
const NUM_LAYERS = parse(Int, get(ENV, "NUM_LAYERS", "4"))
const K_DROP_THRESHOLD = parse(Float64, get(ENV, "K_DROP_THRESHOLD", "0.0"))
const K_DROP_MIN = parse(Int, get(ENV, "K_DROP_MIN", "1"))

const RECON_WEIGHT = parse(Float64, get(ENV, "RECON_WEIGHT", "1.0"))
const KL_WEIGHT = parse(Float64, get(ENV, "KL_WEIGHT", "0.0"))
const PRED_WEIGHT = parse(Float64, get(ENV, "PRED_WEIGHT", "0.3"))
const VAR_DIR_WEIGHT = parse(Float64, get(ENV, "VAR_DIR_WEIGHT", "0.1"))
const VAR_MAG_WEIGHT = parse(Float64, get(ENV, "VAR_MAG_WEIGHT", "0.01"))
const VAR_MAG_LOW = parse(Float64, get(ENV, "VAR_MAG_LOW", "0.2"))
const VAR_MAG_HIGH = parse(Float64, get(ENV, "VAR_MAG_HIGH", "3.5"))
const ENABLE_RECON_TASK = parse(Int, get(ENV, "ENABLE_RECON_TASK", "1"))
const ENABLE_KL_LOSS = parse(Int, get(ENV, "ENABLE_KL_LOSS", "1"))
const ENABLE_PRED_TASK = parse(Int, get(ENV, "ENABLE_PRED_TASK", "1"))
const ENABLE_VAR_DIR_LOSS = parse(Int, get(ENV, "ENABLE_VAR_DIR_LOSS", "1"))
const ENABLE_VAR_MAG_LOSS = parse(Int, get(ENV, "ENABLE_VAR_MAG_LOSS", "1"))
const TEACHER_FORCING = parse(Int, get(ENV, "TEACHER_FORCING", "0"))
const PRED_WARMUP_FRAC = parse(Float64, get(ENV, "PRED_WARMUP_FRAC", "0.1"))
const PRED_START_FRAC = parse(Float64, get(ENV, "PRED_START_FRAC", "0.1"))
const PRED_FULL_FRAC = parse(Float64, get(ENV, "PRED_FULL_FRAC", string(PRED_WARMUP_FRAC)))
const VAR_DIR_START_FRAC = parse(Float64, get(ENV, "VAR_DIR_START_FRAC", "0.6"))
const VAR_DIR_FULL_FRAC = parse(Float64, get(ENV, "VAR_DIR_FULL_FRAC", "0.85"))
const PRED_SELECT_MODE = get(ENV, "PRED_SELECT_MODE", "best")
const USE_PARALLEL = parse(Int, get(ENV, "USE_PARALLEL", "1"))
const DTYPE = get(ENV, "DTYPE", "fp32")
const VAR_LR_BASE_SCALE = parse(Float64, get(ENV, "VAR_LR_BASE_SCALE", "1.0"))
const VAR_LR_START_FRAC = parse(Float64, get(ENV, "VAR_LR_START_FRAC", "0.0"))
const VAR_LR_FULL_FRAC = parse(Float64, get(ENV, "VAR_LR_FULL_FRAC", "0.0"))
const VAR_LR_DECAY_START_FRAC = parse(Float64, get(ENV, "VAR_LR_DECAY_START_FRAC", "1.0"))
const VAR_LR_END_SCALE = parse(Float64, get(ENV, "VAR_LR_END_SCALE", "1.0"))

function parse_bool(v)
    s = lowercase(strip(string(v)))
    return s in ("1", "true", "yes", "on")
end

function resolve_config(cli::Dict{Symbol,Any})
    data_file = string(get(cli, :data_file, DATA_FILE))
    if !isabspath(data_file) && !isfile(data_file) && isfile(joinpath(DATA_DIR, data_file))
        data_file = joinpath(DATA_DIR, data_file)
    end
    meta_file = string(get(cli, :meta_file, META_FILE))
    if isempty(meta_file)
        meta_file = endswith(data_file, ".jld2") ? data_file[1:end-5] * "_meta.jld2" : data_file * "_meta.jld2"
    end
    if !isabspath(meta_file) && !isfile(meta_file) && isfile(joinpath(DATA_DIR, meta_file))
        meta_file = joinpath(DATA_DIR, meta_file)
    end

    checkpoint_dir = string(get(cli, :checkpoint_dir, CHECKPOINT_DIR))
    checkpoint_prefix = string(get(cli, :checkpoint_prefix, CHECKPOINT_PREFIX))
    epochs = parse(Int, string(get(cli, :epochs, EPOCHS)))
    batch_size = parse(Int, string(get(cli, :batch_size, BATCH_SIZE)))
    lr = parse(Float64, string(get(cli, :lr, LR)))
    max_batches = parse(Int, string(get(cli, :max_batches, MAX_BATCHES)))
    save_every = parse(Int, string(get(cli, :save_every, SAVE_EVERY)))
    grad_clip_norm = parse(Float64, string(get(cli, :grad_clip_norm, GRAD_CLIP_NORM)))
    loss_spike_threshold = parse(Float64, string(get(cli, :loss_spike_threshold, LOSS_SPIKE_THRESHOLD)))
    skip_on_spike = parse_bool(get(cli, :skip_on_spike, SKIP_ON_SPIKE))
    warmup_steps = parse(Int, string(get(cli, :warmup_steps, WARMUP_STEPS)))
    dim = parse(Int, string(get(cli, :dim, MODEL_DIM)))
    mamba_d_state = parse(Int, string(get(cli, :mamba_d_state, MAMBA_D_STATE)))
    block_size = parse(Int, string(get(cli, :block_size, BLOCK_SIZE)))
    seq_len = parse(Int, string(get(cli, :seq_len, SEQ_LEN)))
    force_cpu = parse_bool(get(cli, :force_cpu, FORCE_CPU))
    add_timestamp = parse_bool(get(cli, :add_timestamp, ADD_TIMESTAMP))
    use_parallel = parse_bool(get(cli, :use_parallel, USE_PARALLEL))
    dtype = string(get(cli, :dtype, DTYPE))

    k_streams = parse(Int, string(get(cli, :k_streams, K_STREAMS)))
    heads = parse(Int, string(get(cli, :heads, HEADS)))
    num_layers = parse(Int, string(get(cli, :num_layers, NUM_LAYERS)))
    k_drop_threshold = parse(Float64, string(get(cli, :k_drop_threshold, K_DROP_THRESHOLD)))
    k_drop_min_raw = parse(Int, string(get(cli, :k_drop_min, K_DROP_MIN)))
    k_drop_min = max(1, min(k_drop_min_raw, k_streams))

    recon_weight = parse(Float64, string(get(cli, :recon_weight, RECON_WEIGHT)))
    kl_weight = parse(Float64, string(get(cli, :kl_weight, KL_WEIGHT)))
    pred_weight = parse(Float64, string(get(cli, :pred_weight, PRED_WEIGHT)))
    var_dir_weight = parse(Float64, string(get(cli, :var_dir_weight, VAR_DIR_WEIGHT)))
    var_mag_weight = parse(Float64, string(get(cli, :var_mag_weight, VAR_MAG_WEIGHT)))
    var_mag_low = parse(Float64, string(get(cli, :var_mag_low, VAR_MAG_LOW)))
    var_mag_high = parse(Float64, string(get(cli, :var_mag_high, VAR_MAG_HIGH)))
    enable_recon_task = parse_bool(get(cli, :enable_recon_task, ENABLE_RECON_TASK))
    enable_kl_loss = parse_bool(get(cli, :enable_kl_loss, ENABLE_KL_LOSS))
    enable_pred_task = parse_bool(get(cli, :enable_pred_task, ENABLE_PRED_TASK))
    enable_var_dir_loss = parse_bool(get(cli, :enable_var_dir_loss, ENABLE_VAR_DIR_LOSS))
    enable_var_mag_loss = parse_bool(get(cli, :enable_var_mag_loss, ENABLE_VAR_MAG_LOSS))
    teacher_forcing = parse_bool(get(cli, :teacher_forcing, TEACHER_FORCING))
    pred_warmup_frac = clamp(parse(Float64, string(get(cli, :pred_warmup_frac, PRED_WARMUP_FRAC))), 0.0, 1.0)
    pred_start_frac = clamp(parse(Float64, string(get(cli, :pred_start_frac, PRED_START_FRAC))), 0.0, 1.0)
    pred_full_frac = clamp(parse(Float64, string(get(cli, :pred_full_frac, pred_warmup_frac))), 0.0, 1.0)
    var_dir_start_frac = clamp(parse(Float64, string(get(cli, :var_dir_start_frac, VAR_DIR_START_FRAC))), 0.0, 1.0)
    var_dir_full_frac = clamp(parse(Float64, string(get(cli, :var_dir_full_frac, VAR_DIR_FULL_FRAC))), 0.0, 1.0)
    pred_select_mode = lowercase(string(get(cli, :pred_select_mode, PRED_SELECT_MODE)))
    var_lr_base_scale = parse(Float64, string(get(cli, :var_lr_base_scale, VAR_LR_BASE_SCALE)))
    var_lr_start_frac = clamp(parse(Float64, string(get(cli, :var_lr_start_frac, VAR_LR_START_FRAC))), 0.0, 1.0)
    var_lr_full_frac = clamp(parse(Float64, string(get(cli, :var_lr_full_frac, VAR_LR_FULL_FRAC))), 0.0, 1.0)
    var_lr_decay_start_frac = clamp(parse(Float64, string(get(cli, :var_lr_decay_start_frac, VAR_LR_DECAY_START_FRAC))), 0.0, 1.0)
    var_lr_end_scale = parse(Float64, string(get(cli, :var_lr_end_scale, VAR_LR_END_SCALE)))

    if add_timestamp
        ts = Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
        checkpoint_prefix = "$(checkpoint_prefix)_$(ts)"
    end

    return (; data_file, meta_file, checkpoint_dir, checkpoint_prefix, epochs, batch_size, lr, max_batches, save_every, grad_clip_norm, loss_spike_threshold, skip_on_spike, warmup_steps, dim, mamba_d_state, block_size, seq_len, force_cpu, use_parallel, dtype, k_streams, heads, num_layers, k_drop_threshold, k_drop_min, recon_weight, kl_weight, pred_weight, var_dir_weight, var_mag_weight, var_mag_low, var_mag_high, enable_recon_task, enable_kl_loss, enable_pred_task, enable_var_dir_loss, enable_var_mag_loss, teacher_forcing, pred_warmup_frac, pred_start_frac, pred_full_frac, var_dir_start_frac, var_dir_full_frac, pred_select_mode, var_lr_base_scale, var_lr_start_frac, var_lr_full_frac, var_lr_decay_start_frac, var_lr_end_scale)
end

function _phase_ramp(progress::Float32, start_frac::Float32, full_frac::Float32)
    if full_frac <= start_frac + 1f-6
        return progress >= full_frac ? 1f0 : 0f0
    end
    if progress <= start_frac
        return 0f0
    elseif progress >= full_frac
        return 1f0
    else
        return (progress - start_frac) / (full_frac - start_frac)
    end
end

function _phase_decay(progress::Float32, start_frac::Float32, end_scale::Float32)
    if progress <= start_frac
        return 1f0
    end
    t = clamp((progress - start_frac) / max(1f0 - start_frac, 1f-6), 0f0, 1f0)
    return (1f0 - t) + t * end_scale
end

function compute_var_lr_scale(cfg, progress::Float32)
    ramp = _phase_ramp(progress, Float32(cfg.var_lr_start_frac), Float32(cfg.var_lr_full_frac))
    decay = _phase_decay(progress, Float32(cfg.var_lr_decay_start_frac), Float32(cfg.var_lr_end_scale))
    return Float32(cfg.var_lr_base_scale) * ramp * decay
end

function apply_var_grad_scale(grads, scale::Float32)
    if !isfinite(scale) || abs(scale - 1f0) < 1f-6
        return grads
    end
    if !(hasproperty(grads, :model) && hasproperty(grads.model, :encoder) && hasproperty(grads.model.encoder, :var_head))
        return grads
    end
    var_head_scaled = TrainStage1.tree_map(g -> (g isa AbstractArray || g isa Number) ? (g .* scale) : g, grads.model.encoder.var_head)
    encoder_scaled = merge(grads.model.encoder, (var_head=var_head_scaled,))
    model_scaled = merge(grads.model, (encoder=encoder_scaled,))
    return merge(grads, (model=model_scaled,))
end

function apply_precision(ps, st, cfg)
    T = something(TrainStage1.parse_dtype(cfg.dtype), Float32)
    return TrainStage1.cast_floats(ps, T), TrainStage1.cast_floats(st, T), T
end

function select_device(force_cpu::Bool)
    if force_cpu
        return cpu_device()
    end
    return CUDA.functional() ? gpu_device() : cpu_device()
end

function ce_loss_masked(logits, target, pad_id::Int)
    vocab_size = size(logits, 1)
    y_pred_flat = reshape(logits, vocab_size, :)
    y_batch_flat = reshape(target, :)
    logits_flat = eltype(y_pred_flat) <: Union{Float16,Core.BFloat16} ? Float32.(y_pred_flat) : y_pred_flat
    log_probs = TrainStage1.logsoftmax_stable(logits_flat; dims=1)
    mask = y_batch_flat .!= pad_id
    weights = Float32.(mask)
    picked = TrainStage1.gather2d(log_probs, y_batch_flat)
    num = sum((-picked) .* weights)
    den = max(sum(weights), 1f0)
    return num / den
end

function ce_loss_masked_per_batch(logits, target, pad_id::Int)
    B = size(target, 2)
    return ntuple(b -> ce_loss_masked(@view(logits[:, :, b:b]), @view(target[:, b:b]), pad_id), B)
end

function compute_joint_loss(model, mixer, reasoner, ps, st, x_batch, cfg, pad_id::Int, eos_id::Int, pred_weight_eff::Float32, var_dir_weight_eff::Float32)
    capsules_params, st_enc = model.encoder(x_batch, ps.model.encoder, st.model.encoder)
    mu, logvar = capsules_params
    z = Model.reparameterize(mu, logvar; rng=Random.default_rng(), training=true)
    capsules_norm, st_norm = model.norm(z, ps.model.norm, st.model.norm)
    logits_rec, st_dec = model.decoder(capsules_norm, x_batch, ps.model.decoder, st.model.decoder; start_id=eos_id, teacher_forcing=cfg.teacher_forcing)

    recon_loss = ce_loss_masked(logits_rec, x_batch, pad_id)
    logvar_clamped = clamp.(logvar, -10f0, 10f0)
    kl_per_element = -0.5f0 .* (1f0 .+ logvar_clamped .- abs2.(mu) .- exp.(logvar_clamped))
    kl_loss = mean(sum(kl_per_element; dims=1))
    var = exp.(logvar_clamped)
    kl_dim_vector = mean(-0.5f0 .* (1f0 .+ logvar_clamped .- abs2.(mu) .- var); dims=(2,3))

    Dim, Lcap, B = size(mu)
    K = max(cfg.block_size, 1)
    L = size(x_batch, 1)
    stride = max(L ÷ max(Lcap, 1), 1)
    n_pred_tokens = max(min(L - stride, stride * max(Lcap - 1, 0)), 0)

    pred_loss = zero(recon_loss)
    if cfg.enable_pred_task && pred_weight_eff > 0f0 && Lcap > 1 && n_pred_tokens > 0
        x_k_unnorm, st_mix = mixer(capsules_params, cfg.k_streams, ps.mixer, st.mixer)
        x_k_flat = reshape(x_k_unnorm, Dim, :)
        x_k_norm_flat, st_norm2 = model.norm(x_k_flat, ps.model.norm, st.model.norm)
        x_k = reshape(x_k_norm_flat, Dim, cfg.k_streams, Lcap, B)
        y, st_reas = reasoner(x_k, ps.reasoner, st.reasoner)
        target_tokens = copy(@view x_batch[(stride + 1):(stride + n_pred_tokens), :])
        losses_by_k = ntuple(k -> begin
            z_pred_k = copy(@view y[:, k, 1:end-1, :])
            logits_k, _ = model.decoder(z_pred_k, target_tokens, ps.model.decoder, st.model.decoder; start_id=eos_id, teacher_forcing=true)
            ce_loss_masked_per_batch(logits_k, target_tokens, pad_id)
        end, cfg.k_streams)
        if cfg.pred_select_mode == "mean"
            pred_sum = sum(losses_by_k[k][b] for k in 1:cfg.k_streams for b in 1:B)
            pred_loss = pred_sum / max(Float32(cfg.k_streams * B), 1f0)
        else
            best_sum = sum(reduce(min, (losses_by_k[k][b] for k in 1:cfg.k_streams)) for b in 1:B)
            pred_loss = best_sum / max(Float32(B), 1f0)
        end
        st = merge(st, (mixer=st_mix, reasoner=st_reas, model=merge(st.model, (norm=st_norm2,))))
    end

    var_dir_loss = zero(recon_loss)
    if cfg.enable_var_dir_loss && var_dir_weight_eff > 0f0 && Lcap > 1
        eos_mask_tokens = (x_batch .== eos_id)
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

    abs_logvar = abs.(logvar_clamped)
    high_penalty = NNlib.relu.(abs_logvar .- Float32(cfg.var_mag_high))
    low_penalty = NNlib.relu.(Float32(cfg.var_mag_low) .- abs_logvar)
    var_mag_loss = mean(abs2, high_penalty) + mean(abs2, low_penalty)
    mean_var = mean(exp.(logvar_clamped))
    z_mean_val = mean(mu)
    z_std_val = mean(exp.(0.5f0 .* logvar_clamped))

    total_loss = 0f0
    total_loss += cfg.enable_recon_task ? Float32(cfg.recon_weight) * recon_loss : 0f0
    total_loss += cfg.enable_kl_loss ? Float32(cfg.kl_weight) * kl_loss : 0f0
    total_loss += cfg.enable_pred_task ? pred_weight_eff * pred_loss : 0f0
    total_loss += cfg.enable_var_dir_loss ? var_dir_weight_eff * var_dir_loss : 0f0
    total_loss += cfg.enable_var_mag_loss ? Float32(cfg.var_mag_weight) * var_mag_loss : 0f0

    st_out = merge(st, (model=merge(st.model, (encoder=st_enc, norm=st_norm, decoder=st_dec,)),))
    return total_loss, st_out, (; recon=recon_loss, kl=kl_loss, pred=pred_loss, var_dir=var_dir_loss, var_mag=var_mag_loss, var=mean_var, abs_logvar=mean(abs_logvar), z_mean=z_mean_val, z_std=z_std_val, kl_dim=Zygote.dropgrad(kl_dim_vector))
end

function train(cfg)
    mkpath(cfg.checkpoint_dir)
    dev = select_device(cfg.force_cpu)
    cpu = cpu_device()
    println("Using device: $dev")
    enable_bf16_gpu_fallback = CUDA.functional() && !cfg.force_cpu && lowercase(cfg.dtype) in ("bf16", "bfloat16")
    data_content, reset_flags, vocab_size, is_stream, is_bucketed, pad_id, eos_id, _mask_id, _vocab, meta_data = TrainStage1.load_data(cfg.data_file, cfg.meta_file)
    N_total = is_stream ? length(data_content) : (is_bucketed ? sum(size(v, 2) for (_, v) in data_content) : size(data_content, 2))
    println("Loaded data. Total: $N_total, Vocab: $vocab_size")
    flush(stdout)

    model = HMTR_Stage1_AutoEncoder(vocab_size, cfg.dim; block_size=cfg.block_size, pad_id=pad_id, eos_id=eos_id, mamba_d_state=cfg.mamba_d_state, use_parallel=cfg.use_parallel)
    mixer = InitialMixingLayer(cfg.dim, cfg.k_streams)
    reasoner = MHCLatentReasoner(cfg.dim, cfg.k_streams, cfg.heads, cfg.num_layers; k_drop_threshold=Float32(cfg.k_drop_threshold), k_drop_min=cfg.k_drop_min)
    rng = Random.default_rng()
    Random.seed!(rng, 42)
    ps_model, st_model = Lux.setup(rng, model)
    ps_mix, st_mix = Lux.setup(rng, mixer)
    ps_reas, st_reas = Lux.setup(rng, reasoner)
    ps, st, dtype_used = apply_precision((model=ps_model, mixer=ps_mix, reasoner=ps_reas), (model=st_model, mixer=st_mix, reasoner=st_reas), cfg)
    println("Precision | dtype=$(dtype_used)")
    ps = ps |> dev
    st = st |> dev
    if enable_bf16_gpu_fallback
        PrecisionRuntime.install_bf16_gpu_dense_fallback!()
        println("Precision Runtime | BF16 GPU dense fallback enabled")
    end

    opt = Optimisers.Adam(cfg.lr)
    opt_state = Optimisers.setup(opt, ps)
    train_step = 0

    local num_batches_per_epoch
    if is_stream
        stream_iter = TrainStage1.StreamBatchIterator(data_content, reset_flags, cfg.batch_size, cfg.seq_len, cfg.block_size, eos_id, pad_id)
        num_batches_per_epoch = cfg.max_batches > 0 ? min(length(stream_iter), cfg.max_batches) : length(stream_iter)
    else
        batches = TrainStage1.make_batches_impl(data_content, is_stream, is_bucketed, cfg.batch_size, rng)
        num_batches_per_epoch = cfg.max_batches > 0 ? min(length(batches), cfg.max_batches) : length(batches)
    end
    total_steps = cfg.epochs * max(num_batches_per_epoch, 1)
    println("Training Plan: $total_steps total steps, $(cfg.warmup_steps) warmup steps.")
    flush(stdout)

    for epoch in 1:cfg.epochs
        total_loss = 0.0
        n_updates = 0
        batches = if is_stream
            it = TrainStage1.StreamBatchIterator(data_content, reset_flags, cfg.batch_size, cfg.seq_len, cfg.block_size, eos_id, pad_id)
            cfg.max_batches > 0 ? Iterators.take(it, cfg.max_batches) : it
        else
            b = TrainStage1.make_batches_impl(data_content, is_stream, is_bucketed, cfg.batch_size, rng)
            cfg.max_batches > 0 ? b[1:min(length(b), cfg.max_batches)] : b
        end

        for (i, batch_data) in enumerate(batches)
            local x_cpu_raw, reset_mask
            if is_stream
                x_cpu_raw, reset_mask = batch_data
                x_cpu_raw = Matrix{Int}(x_cpu_raw)
            else
                b_key, b_ids = batch_data
                tokens_cpu = is_bucketed ? data_content[b_key][:, b_ids] : data_content[:, b_ids]
                size(tokens_cpu, 2) == 0 && continue
                x_cpu_raw = Matrix{Int}(tokens_cpu)
                reset_mask = trues(size(x_cpu_raw, 2))
            end

            target_lr = TrainStage1.get_target_lr(cfg.lr)
            current_lr = TrainStage1.get_lr(train_step + 1, cfg.warmup_steps, target_lr, total_steps)
            Optimisers.adjust!(opt_state, current_lr)
            progress = Float32(train_step) / max(Float32(total_steps), 1f0)
            pred_ramp = _phase_ramp(progress, Float32(cfg.pred_start_frac), Float32(cfg.pred_full_frac))
            pred_weight_eff = Float32(cfg.pred_weight) * pred_ramp
            var_dir_ramp = _phase_ramp(progress, Float32(cfg.var_dir_start_frac), Float32(cfg.var_dir_full_frac))
            var_dir_weight_eff = Float32(cfg.var_dir_weight) * var_dir_ramp
            var_lr_scale = compute_var_lr_scale(cfg, progress)

            TrainStage1.apply_reset!(st, reset_mask)
            x_batch = x_cpu_raw |> dev
            st_holder = Ref{Any}(nothing)
            internals_holder = Ref{Any}(nothing)
            loss, grad_tuple = Zygote.withgradient(ps) do p
                l, st_tmp, internals_tmp = compute_joint_loss(model, mixer, reasoner, p, st, x_batch, cfg, pad_id, eos_id, pred_weight_eff, var_dir_weight_eff)
                st_holder[] = st_tmp
                internals_holder[] = internals_tmp
                return l
            end
            st_new = st_holder[]
            internals = internals_holder[]
            grads = grad_tuple[1]
            loss_val = Float32(loss)

            spike = !isfinite(loss_val) || loss_val > Float32(cfg.loss_spike_threshold)
            if spike
                stats = TrainStage1.batch_stats(x_batch, vocab_size, pad_id, eos_id)
                pad_frac = stats.n_pad / max(stats.n_total, 1)
                eos_frac = stats.n_eos / max(stats.n_total, 1)
                @printf "SPIKE Epoch %d Step %d Loss %.4f | x[min=%d max=%d] pad=%.3f eos=%.3f\n" epoch i loss_val stats.x_min stats.x_max pad_frac eos_frac
                if cfg.skip_on_spike
                    continue
                end
            end

            grads, grad_norm, _grad_scale = TrainStage1.clip_grads(grads, Float32(cfg.grad_clip_norm))
            grads = apply_var_grad_scale(grads, var_lr_scale)
            opt_state, ps = Optimisers.update(opt_state, ps, grads)
            st = st_new
            total_loss += loss_val
            n_updates += 1
            train_step += 1

            if i % 20 == 0
                au = count(internals.kl_dim .> 0.01f0)
                @printf "Epoch %d [%d/%d] Loss: %.4f (Recon: %.4f | Pred: %.4f | VarDir: %.4f | VarMag: %.4f | KL: %.4f) | w[r=%.3f p=%.3f vd=%.3f vm=%.3f kl=%.3f] | ramp[p=%.2f vd=%.2f] | Var: %.4f | |logvar|: %.4f | z_μ: %.4f | z_σ: %.4f | AU: %d | |g|: %.4f | LR: %.2e | VarLRx: %.3f\n" epoch i num_batches_per_epoch loss_val internals.recon internals.pred internals.var_dir internals.var_mag internals.kl cfg.recon_weight pred_weight_eff var_dir_weight_eff cfg.var_mag_weight cfg.kl_weight pred_ramp var_dir_ramp internals.var internals.abs_logvar internals.z_mean internals.z_std au grad_norm current_lr var_lr_scale
            end

            if cfg.save_every > 0 && (i % cfg.save_every == 0)
                jldsave(joinpath(cfg.checkpoint_dir, "$(cfg.checkpoint_prefix)_epoch$(epoch)_step$(i).jld2"); ps=ps |> cpu, st=st |> cpu, opt_state=opt_state |> cpu, epoch=epoch, step=i, train_step=train_step, config=cfg, meta_data=meta_data)
            end
        end

        avg_loss = total_loss / max(n_updates, 1)
        println("Epoch $epoch Completed. Avg Loss: $avg_loss")
        flush(stdout)
        jldsave(joinpath(cfg.checkpoint_dir, "$(cfg.checkpoint_prefix)_epoch$epoch.jld2"); ps=ps |> cpu, st=st |> cpu, opt_state=opt_state |> cpu, epoch=epoch, step=0, train_step=train_step, config=cfg, meta_data=meta_data)
    end
end

function train_stage_joint(args::Vector{String})
    if "--help" in args || "-h" in args
        println("Usage: train_stage_joint [options]")
        println("Options:")
        println("  --data-file <path>         Path to processed data file")
        println("  --meta-file <path>         Path to metadata file")
        println("  --checkpoint-dir <path>    Directory to save checkpoints")
        println("  --checkpoint-prefix <str>  Checkpoint prefix")
        println("  --epochs <int>             Number of epochs")
        println("  --batch-size <int>         Batch size")
        println("  --lr <float>               Learning rate")
        println("  --dim <int>                Model dimension")
        println("  --mamba-d-state <int>      Mamba d_state")
        println("  --block-size <int>         Capsule block size")
        println("  --seq-len <int>            Stream chunk length")
        println("  --k-streams <int>          Number of reasoner streams")
        println("  --heads <int>              Reasoner attention heads")
        println("  --num-layers <int>         Reasoner layers")
        println("  --recon-weight <float>     Reconstruction loss weight")
        println("  --pred-weight <float>      Prediction loss weight")
        println("  --kl-weight <float>        KL loss weight")
        println("  --var-dir-weight <float>   Variance direction loss weight")
        println("  --var-mag-weight <float>   Variance magnitude loss weight")
        println("  --enable-recon-task <0|1>  Enable reconstruction task")
        println("  --enable-pred-task <0|1>   Enable prediction task")
        println("  --enable-kl-loss <0|1>     Enable KL loss")
        println("  --enable-var-dir-loss <0|1> Enable variance direction loss")
        println("  --enable-var-mag-loss <0|1> Enable variance magnitude loss")
        println("  --pred-select-mode <best|mean> Route selection for prediction loss")
        println("  --pred-warmup-frac <0-1>   Warmup fraction for prediction loss")
        println("  --pred-start-frac <0-1>    Start fraction for prediction curriculum ramp")
        println("  --pred-full-frac <0-1>     Full fraction for prediction curriculum ramp")
        println("  --var-dir-start-frac <0-1> Start fraction for variance-direction curriculum ramp")
        println("  --var-dir-full-frac <0-1>  Full fraction for variance-direction curriculum ramp")
        println("  --var-lr-base-scale <float> Base LR multiplier for var_head gradients")
        println("  --var-lr-start-frac <0-1>  Start fraction for var LR ramp")
        println("  --var-lr-full-frac <0-1>   Full fraction for var LR ramp")
        println("  --var-lr-decay-start-frac <0-1> Start fraction for var LR decay")
        println("  --var-lr-end-scale <float> End LR multiplier for var_head gradients")
        println("  --teacher-forcing <0|1>    Use teacher forcing for reconstruction")
        println("  --dtype <fp32|fp16|bf16>   Precision for model parameters/states")
        println("  --use-parallel <0|1>       Enable parallel scan")
        println("  --force-cpu <0|1>          Force CPU training")
        return
    end
    cli = Utils.parse_cli_args(args)
    cfg = resolve_config(cli)
    train(cfg)
end

end
