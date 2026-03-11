module TrainStage2

using Lux
using Random
using JLD2
using Optimisers
using Zygote
using Printf
using CUDA
using LuxCUDA
using Dates
using Statistics
using ..Utils
using ..Model
import ..TrainStage1

export train_stage2

const PROJECT_ROOT = normpath(joinpath(@__DIR__, ".."))
const DATA_DIR = get(ENV, "DATA_DIR", joinpath(PROJECT_ROOT, "data"))
const DATA_FILE = get(ENV, "DATA_FILE", joinpath(DATA_DIR, "processed.jld2"))
const META_FILE = get(ENV, "META_FILE", "")
const CHECKPOINT_DIR = get(ENV, "CHECKPOINT_DIR", joinpath(PROJECT_ROOT, "checkpoints"))
const CHECKPOINT_PREFIX = get(ENV, "CHECKPOINT_PREFIX", "ckpt_stage2")
const STAGE1_CKPT = get(ENV, "STAGE1_CKPT", "")
const RESUME_CKPT = get(ENV, "RESUME_CKPT", "")
const EPOCHS = parse(Int, get(ENV, "EPOCHS", "5"))
const BATCH_SIZE = parse(Int, get(ENV, "BATCH_SIZE", "32"))
const LR = parse(Float64, get(ENV, "LR", "1e-4"))
const MAX_BATCHES = parse(Int, get(ENV, "MAX_BATCHES", "0"))
const SAVE_EVERY = parse(Int, get(ENV, "SAVE_EVERY", "0"))
const ADD_TIMESTAMP = parse(Int, get(ENV, "ADD_TIMESTAMP", "1"))
const GRAD_CLIP_NORM = parse(Float64, get(ENV, "GRAD_CLIP_NORM", "5.0"))
const LOSS_SPIKE_THRESHOLD = parse(Float64, get(ENV, "LOSS_SPIKE_THRESHOLD", "10.0"))
const SKIP_ON_SPIKE = parse(Int, get(ENV, "SKIP_ON_SPIKE", "1"))
const K_DROP_THRESHOLD = parse(Float64, get(ENV, "K_DROP_THRESHOLD", "0.0"))
const K_DROP_MIN = parse(Int, get(ENV, "K_DROP_MIN", "1"))
const WARMUP_STEPS = parse(Int, get(ENV, "WARMUP_STEPS", "500"))
const FORCE_CPU = parse(Int, get(ENV, "FORCE_CPU", "0"))
const K_STREAMS = parse(Int, get(ENV, "K_STREAMS", "4"))
const HEADS = parse(Int, get(ENV, "HEADS", "8"))
const NUM_LAYERS = parse(Int, get(ENV, "NUM_LAYERS", "4"))
const BLOCK_SIZE = parse(Int, get(ENV, "BLOCK_SIZE", "8"))
const SEQ_LEN = parse(Int, get(ENV, "SEQ_LEN", "1024"))

function parse_bool(key, env_default_int)
    val = string(get(ENV, key, env_default_int))
    return val == "true" || val == "1"
end

function resolve_config(cli::Dict{Symbol, Any})
    data_file = get(cli, :data_file, DATA_FILE)
    if !isabspath(data_file)
        if !isfile(data_file) && isfile(joinpath(DATA_DIR, data_file))
            data_file = joinpath(DATA_DIR, data_file)
        end
    end
    meta_file = string(get(cli, :meta_file, META_FILE))
    if isempty(meta_file)
        if endswith(data_file, ".jld2")
            meta_file = data_file[1:end - 5] * "_meta.jld2"
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
    stage1_ckpt = string(get(cli, :stage1_ckpt, STAGE1_CKPT))
    resume_ckpt = string(get(cli, :resume_ckpt, get(cli, :resume, get(cli, :resume_from, RESUME_CKPT))))
    epochs = parse(Int, string(get(cli, :epochs, EPOCHS)))
    batch_size = parse(Int, string(get(cli, :batch_size, BATCH_SIZE)))
    lr = parse(Float64, string(get(cli, :lr, LR)))
    max_batches = parse(Int, string(get(cli, :max_batches, MAX_BATCHES)))
    save_every = parse(Int, string(get(cli, :save_every, SAVE_EVERY)))
    warmup_steps = parse(Int, string(get(cli, :warmup_steps, WARMUP_STEPS)))
    k_streams = parse(Int, string(get(cli, :k_streams, K_STREAMS)))
    heads = parse(Int, string(get(cli, :heads, HEADS)))
    num_layers = parse(Int, string(get(cli, :num_layers, NUM_LAYERS)))
    block_size = parse(Int, string(get(cli, :block_size, BLOCK_SIZE)))
    seq_len = parse(Int, string(get(cli, :seq_len, SEQ_LEN)))

    skip_on_spike = haskey(cli, :skip_on_spike) ? string(get(cli, :skip_on_spike, "0")) in ("true", "1") : parse_bool("SKIP_ON_SPIKE", SKIP_ON_SPIKE)
    use_parallel = haskey(cli, :use_parallel) ? string(get(cli, :use_parallel, "0")) in ("true", "1") : false
    force_cpu = string(get(cli, :force_cpu, FORCE_CPU)) in ("true", "1")
    add_timestamp = haskey(cli, :add_timestamp) ? string(get(cli, :add_timestamp, "1")) in ("true", "1") : string(get(ENV, "ADD_TIMESTAMP", ADD_TIMESTAMP)) in ("true", "1")

    if !isempty(resume_ckpt) && !isfile(resume_ckpt)
        alt_path = joinpath(checkpoint_dir, basename(resume_ckpt))
        if isfile(alt_path)
            resume_ckpt = alt_path
        elseif isfile(joinpath(checkpoint_dir, resume_ckpt))
            resume_ckpt = joinpath(checkpoint_dir, resume_ckpt)
        end
    end

    if add_timestamp
        ts = Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
        checkpoint_prefix = "$(checkpoint_prefix)_$(ts)"
    end

    grad_clip_norm = parse(Float64, string(get(cli, :grad_clip_norm, GRAD_CLIP_NORM)))
    loss_spike_threshold = parse(Float64, string(get(cli, :loss_spike_threshold, LOSS_SPIKE_THRESHOLD)))
    k_drop_threshold = parse(Float64, string(get(cli, :k_drop_threshold, K_DROP_THRESHOLD)))
    k_drop_min_raw = parse(Int, string(get(cli, :k_drop_min, K_DROP_MIN)))
    k_drop_min = max(1, min(k_drop_min_raw, k_streams))

    return (; data_file, meta_file, checkpoint_dir, checkpoint_prefix, stage1_ckpt, resume_ckpt, epochs, batch_size, lr, max_batches, save_every, warmup_steps, k_streams, heads, num_layers, block_size, seq_len, skip_on_spike, force_cpu, grad_clip_norm, loss_spike_threshold, k_drop_threshold, k_drop_min, use_parallel)
end

function select_device(force_cpu::Bool)
    if force_cpu
        return cpu_device()
    end
    return CUDA.functional() ? gpu_device() : cpu_device()
end

function infer_mamba_d_state_from_ps(ps, dim::Int)
    function find_d_state(x)
        if x isa NamedTuple
            if haskey(x, :adt_proj)
                ap = x.adt_proj
                if ap isa NamedTuple && haskey(ap, :weight) && ap.weight isa AbstractArray
                    out_dim = size(ap.weight, 1)
                    diff = out_dim - dim
                    if diff > 0 && diff % 2 == 0
                        return diff ÷ 2
                    end
                end
            end
            for v in values(x)
                ds = find_d_state(v)
                ds === nothing || return ds
            end
            return nothing
        elseif x isa Tuple
            for v in x
                ds = find_d_state(v)
                ds === nothing || return ds
            end
            return nothing
        else
            return nothing
        end
    end
    return find_d_state(ps)
end

function get_target_lr(base_lr::Float64)
    return base_lr
end

function load_stage1(cfg, vocab_size::Int, pad_id::Int, eos_id::Int)
    if isempty(cfg.stage1_ckpt) || !isfile(cfg.stage1_ckpt)
        error("stage1_ckpt=$(cfg.stage1_ckpt) not found")
    end
    ckpt = JLD2.load(cfg.stage1_ckpt)
    ps = ckpt["ps"]
    st = ckpt["st"]
    dim = size(ps.encoder.embedding.weight, 1)
    vocab_ckpt = size(ps.encoder.embedding.weight, 2)
    mamba_d_state = if haskey(ckpt, "mamba_d_state")
        Int(ckpt["mamba_d_state"])
    else
        ds = infer_mamba_d_state_from_ps(ps, dim)
        ds === nothing ? 16 : Int(ds)
    end
    model = HMTR_Stage1_AutoEncoder(vocab_size, dim; block_size=cfg.block_size, pad_id=pad_id, eos_id=eos_id, mamba_d_state=mamba_d_state, use_parallel=cfg.use_parallel)
    rng = Random.default_rng()
    Random.seed!(rng, 42)
    ps0, st0 = Lux.setup(rng, model)
    
    if haskey(ps.encoder, :head)
        # Legacy checkpoint transition
        old_head_w = ps.encoder.head.weight
        old_head_b = ps.encoder.head.bias
        mu_w = old_head_w[1:dim, :]
        var_w = old_head_w[dim+1:end, :]
        mu_b = old_head_b[1:dim]
        var_b = old_head_b[dim+1:end]
        
        encoder_new = merge(ps.encoder, (
            mu_head=(weight=mu_w, bias=mu_b),
            var_head=(weight=var_w, bias=var_b)
        ))
        
        # Remove the old :head from NamedTuple
        keys_keep = filter(k -> k != :head, keys(encoder_new))
        encoder_clean = NamedTuple{keys_keep}(Tuple(encoder_new[k] for k in keys_keep))
        
        ps = merge(ps, (encoder=encoder_clean,))
        
        # Also clean state
        st_enc_new = merge(st.encoder, (
            mu_head=NamedTuple(),
            var_head=NamedTuple()
        ))
        keys_keep_st = filter(k -> k != :head, keys(st_enc_new))
        st_enc_clean = NamedTuple{keys_keep_st}(Tuple(st_enc_new[k] for k in keys_keep_st))
        st = merge(st, (encoder=st_enc_clean,))
    end

    if vocab_ckpt != vocab_size
        println("Warning: Vocab size mismatch! Resizing from $(vocab_ckpt) to $(vocab_size).")
        n_copy = min(vocab_ckpt, vocab_size)
        
        # Resize encoder embedding
        new_enc_embed = copy(ps0.encoder.embedding.weight)
        new_enc_embed[:, 1:n_copy] .= ps.encoder.embedding.weight[:, 1:n_copy]
        ps = merge(ps, (encoder=merge(ps.encoder, (embedding=(weight=new_enc_embed,),)),))
        
        # Resize decoder if exists
        if haskey(ps, :decoder)
            new_dec_embed = copy(ps0.decoder.embedding.weight)
            if haskey(ps.decoder, :embedding)
                new_dec_embed[:, 1:n_copy] .= ps.decoder.embedding.weight[:, 1:n_copy]
            end
            
            new_dec_proj_w = copy(ps0.decoder.proj.weight)
            new_dec_proj_b = copy(ps0.decoder.proj.bias)
            if haskey(ps.decoder, :proj)
                new_dec_proj_w[1:n_copy, :] .= ps.decoder.proj.weight[1:n_copy, :]
                
                # Bias could be [V, 1] or [V]
                if length(size(ps.decoder.proj.bias)) >= 2
                    new_dec_proj_b[1:n_copy, 1] .= ps.decoder.proj.bias[1:n_copy, 1]
                else
                    new_dec_proj_b[1:n_copy] .= ps.decoder.proj.bias[1:n_copy]
                end
            end
            
            ps = merge(ps, (decoder=merge(ps.decoder, (embedding=(weight=new_dec_embed,), proj=(weight=new_dec_proj_w, bias=new_dec_proj_b))),))
        end
    end
    
    if !haskey(ps, :norm)
        ps = merge(ps, (norm=ps0.norm,))
    end
    if !haskey(st, :norm)
        st = merge(st, (norm=st0.norm,))
    end
    if !haskey(st, :encoder)
        st = merge(st, (encoder=st0.encoder,))
    end
    return model, ps, st, dim, mamba_d_state
end

function compute_loss(model1, mixer, reasoner, ps_train, st, x_batch, eos_id, pad_id, K_dyn::Int, kl_weight::Float32, recon_weight::Float32)
    # Reconstruct parameter tree for full Mamba encode (Zygote friendly)
    encoder_ps = (embedding=st.frozen_ps.embedding, layers=st.frozen_ps.layers, mu_head=st.frozen_ps.mu_head, var_head=ps_train.var_head)
    encoder_st = (embedding=st.encoder.embedding, layers=st.encoder.layers, mu_head=st.encoder.mu_head, var_head=st.var_head)

    # 1. Encode
    capsules_params, st_enc = model1.encoder(x_batch, encoder_ps, encoder_st)
    mu, logvar = capsules_params
    mu = Zygote.dropgrad(mu) # Manifold Preservation Requirement
    
    Dim, Lcap, B = size(mu)
    
    # KL Divergence
    kl_div = -0.5f0 * sum(1f0 .+ logvar .- abs2.(mu) .- exp.(logvar)) / max(Lcap * B, 1)

    rng = Random.default_rng()
    z = Model.reparameterize(mu, logvar; rng=rng, training=true)
    capsules_norm, _st_norm_caps = model1.norm(z, ps_train.norm, st.norm)
    logits, _st_dec = model1.decoder(capsules_norm, x_batch, st.frozen_dec, st.decoder; start_id=eos_id, teacher_forcing=true)
    vocab_size = size(logits, 1)
    L = size(x_batch, 1)
    K = max(getfield(model1.decoder, :block_size), 1)
    Lpad = K * cld(L, K)
    y_pred_pad = if Lpad == L
        logits
    else
        pad_part = similar(logits, vocab_size, Lpad - L, B)
        Zygote.@ignore fill!(pad_part, zero(eltype(pad_part)))
        cat(logits, pad_part; dims=2)
    end
    y_batch_pad = if Lpad == L
        x_batch
    else
        pad_part = similar(x_batch, Lpad - L, B)
        Zygote.@ignore fill!(pad_part, pad_id)
        cat(x_batch, pad_part; dims=1)
    end
    y_pred_flat = reshape(y_pred_pad, vocab_size, :)
    y_batch_flat = reshape(y_batch_pad, :)
    logits_flat = eltype(y_pred_flat) <: Union{Float16,Core.BFloat16} ? Float32.(y_pred_flat) : y_pred_flat
    log_probs = TrainStage1.logsoftmax_stable(logits_flat; dims=1)
    mask = y_batch_flat .!= pad_id
    weights = Float32.(mask)
    picked = TrainStage1.gather2d(log_probs, y_batch_flat)
    recon_loss = sum((-picked) .* weights) / max(sum(weights), 1)

    # 2. Initial Mixing
    x_k_unnorm, st_mix = mixer(capsules_params, K_dyn, ps_train.mixer, st.mixer)

    # 3. Normalization
    x_k_flat = reshape(x_k_unnorm, Dim, :)
    x_k_norm_flat, st_norm = model1.norm(x_k_flat, ps_train.norm, st.norm)
    x_k = reshape(x_k_norm_flat, Dim, K_dyn, Lcap, B)

    # 4. Reasoner
    y, st_reas = reasoner(x_k, ps_train.reasoner, st.reasoner)
    y_mean = dropdims(mean(y; dims=2); dims=2)
    
    if Lcap <= 1
        pred_loss = zero(eltype(y_mean))
        loss = pred_loss + kl_weight * kl_div + recon_weight * recon_loss
        return loss, (var_head=st_enc.var_head, norm=st_norm, mixer=st_mix, reasoner=st_reas), (; pred=pred_loss, kl=kl_div, recon=recon_loss, lcap=Lcap)
    end
    
    # 5. Target
    mu_flat = reshape(mu, Dim, :)
    mu_norm_flat, _ = model1.norm(mu_flat, ps_train.norm, st.norm)
    mu_norm = reshape(mu_norm_flat, Dim, Lcap, B)
    target = Zygote.dropgrad(@view mu_norm[:, 2:end, :])
    pred = @view y_mean[:, 1:end-1, :]
    
    L = size(x_batch, 1)
    stride = L ÷ Lcap
    eos_mask_tokens = (x_batch .== eos_id)
    eos_mask_4d = reshape(eos_mask_tokens, stride, Lcap, B)
    capsule_has_eos = dropdims(any(eos_mask_4d; dims=1); dims=1)
    valid_transition = .!(capsule_has_eos[1:end-1, :])

    pred_normsq = sum(abs2, pred; dims=1) .+ Float32(1e-6)
    target_normsq = sum(abs2, target; dims=1) .+ Float32(1e-6)
    dot_product = sum(pred .* target; dims=1)
    cos_sim = dot_product ./ sqrt.(pred_normsq .* target_normsq)
    
    per_transition_loss = Float32(1.0) .- cos_sim
    masked_loss = per_transition_loss .* reshape(valid_transition, 1, Lcap-1, B)
    num_valid = sum(valid_transition)
    
    pred_loss = sum(masked_loss) / max(num_valid, 1)
    loss = pred_loss + kl_weight * kl_div + recon_weight * recon_loss
    
    return loss, (var_head=st_enc.var_head, norm=st_norm, mixer=st_mix, reasoner=st_reas), (; pred=pred_loss, kl=kl_div, recon=recon_loss, lcap=Lcap)
end

function train(cfg)
    mkpath(cfg.checkpoint_dir)
    dev = select_device(cfg.force_cpu)
    cpu = cpu_device()
    println("Using device: $dev")
    flush(stdout)

    data_content, reset_flags, vocab_size, is_stream, is_bucketed, pad_id, eos_id, _mask_id, _vocab = TrainStage1.load_data(cfg.data_file, cfg.meta_file)
    N_total = 0
    if is_stream
        N_total = length(data_content)
        println("Loaded STREAM data. Total tokens: $N_total, Vocab: $vocab_size")
    elseif is_bucketed
        for (_, v) in data_content
            N_total += size(v, 2)
        end
        println("Loaded bucketed data. Total samples: $N_total, Vocab: $vocab_size.")
    else
        N_total = size(data_content, 2)
        println("Loaded simple data. Total samples: $N_total, Vocab: $vocab_size")
    end
    flush(stdout)

    model1, ps1, st1, dim, _mamba_d_state = load_stage1(cfg, vocab_size, pad_id, eos_id)
    ps1 = ps1 |> dev
    st1 = st1 |> dev

    rng = Random.default_rng()
    Random.seed!(rng, 42)
    mixer = InitialMixingLayer(dim, cfg.k_streams)
    reasoner = MHCLatentReasoner(dim, cfg.k_streams, cfg.heads, cfg.num_layers; k_drop_threshold=Float32(cfg.k_drop_threshold), k_drop_min=cfg.k_drop_min)
    ps_mix, st_mix = Lux.setup(rng, mixer)
    ps_reas, st_reas = Lux.setup(rng, reasoner)
    
    # Isolate var_head for Stage 2 optimization
    ps_train = (; var_head=ps1.encoder.var_head, norm=ps1.norm, mixer=ps_mix, reasoner=ps_reas)
    st_train = (; var_head=st1.encoder.var_head, norm=st1.norm, mixer=st_mix, reasoner=st_reas, frozen_ps=(embedding=ps1.encoder.embedding, layers=ps1.encoder.layers, mu_head=ps1.encoder.mu_head), frozen_dec=ps1.decoder, encoder=st1.encoder, decoder=st1.decoder)

    resume_opt_state = nothing
    train_step = 0
    if !isempty(cfg.resume_ckpt)
        if !isfile(cfg.resume_ckpt)
            error("resume_ckpt=$(cfg.resume_ckpt) not found")
        end
        ckpt = JLD2.load(cfg.resume_ckpt)
        ps_train = ckpt["ps"]
        st_train = ckpt["st"]
        if haskey(ckpt, "opt_state")
            resume_opt_state = ckpt["opt_state"]
        end
        if haskey(ckpt, "train_step")
            train_step = Int(ckpt["train_step"])
        end
        println("Resuming from ckpt=$(cfg.resume_ckpt) train_step=$(train_step)")
    end
    if !haskey(st_train, :frozen_dec)
        st_train = merge(st_train, (frozen_dec=ps1.decoder, decoder=st1.decoder))
    end

    ps_train = ps_train |> dev
    st_train = st_train |> dev

    opt = Optimisers.Adam(cfg.lr)
    opt_state = Optimisers.setup(opt, ps_train)
    if resume_opt_state !== nothing
        opt_state = resume_opt_state |> dev
    end

    local batches
    local num_batches_per_epoch
    local stream_iter
    if is_stream
        stream_iter = TrainStage1.StreamBatchIterator(data_content, reset_flags, cfg.batch_size, cfg.seq_len)
        if cfg.max_batches > 0
            batches = Iterators.take(stream_iter, cfg.max_batches)
            num_batches_per_epoch = min(length(stream_iter), cfg.max_batches)
        else
            batches = stream_iter
            num_batches_per_epoch = length(stream_iter)
        end
    else
        batches = TrainStage1.make_batches_impl(data_content, is_stream, is_bucketed, cfg.batch_size, rng)
        if cfg.max_batches > 0
            batches = batches[1:min(length(batches), cfg.max_batches)]
        end
        num_batches_per_epoch = length(batches)
    end

    warmup_steps = cfg.warmup_steps
    total_steps = cfg.epochs * num_batches_per_epoch
    println("Training Plan: $total_steps total steps, $warmup_steps warmup steps.")
    flush(stdout)

    println("Starting training loop... (First batch may take time to compile)")
    flush(stdout)

    for epoch in 1:cfg.epochs
        total_loss = 0.0
        n_updates = 0

        if !is_stream
            batches = TrainStage1.make_batches_impl(data_content, is_stream, is_bucketed, cfg.batch_size, rng)
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
                reset_mask = falses(size(x_cpu_raw, 2))
            end

            target_lr = get_target_lr(cfg.lr)
            current_lr = TrainStage1.get_lr(train_step + 1, warmup_steps, target_lr, total_steps)
            Optimisers.adjust!(opt_state, current_lr)

            TrainStage1.apply_reset!(st_train, reset_mask)

            x_batch = x_cpu_raw |> dev
            K_dyn = rand(1:cfg.k_streams)
            kl_weight = Float32(parse(Float64, get(ENV, "KL_WEIGHT", "0.01")))
            recon_weight = Float32(parse(Float64, get(ENV, "RECON_WEIGHT", "1.0")))
            
            (loss, st_new, internals), back = Zygote.pullback(
                p -> compute_loss(model1, mixer, reasoner, p, st_train, x_batch, eos_id, pad_id, K_dyn, kl_weight, recon_weight), ps_train
            )

            loss_val = Float32(loss)
            spike = !isfinite(loss_val) || loss_val > Float32(cfg.loss_spike_threshold)
            if spike
                stats = TrainStage1.batch_stats(x_batch, vocab_size, pad_id, eos_id)
                pad_frac = stats.n_pad / max(stats.n_total, 1)
                eos_frac = stats.n_eos / max(stats.n_total, 1)
                @printf "SPIKE Epoch %d Step %d Loss %.4f | x[min=%d max=%d] pad=%.3f eos=%.3f\n" epoch i loss_val stats.x_min stats.x_max pad_frac eos_frac
                if !isfinite(loss_val)
                    st_train = st_new
                end
                if cfg.skip_on_spike
                    continue
                end
            end

            grads = back((one(loss), nothing, nothing))[1]
            grads, grad_norm, grad_scale = TrainStage1.clip_grads(grads, Float32(cfg.grad_clip_norm))
            if grad_scale < 1f0 && (i % 50 == 0 || grad_scale < 0.2f0)
                @printf "GradClip Epoch %d Step %d | norm=%.4f scale=%.6f\n" epoch i grad_norm grad_scale
            end

            opt_state, ps_train = Optimisers.update(opt_state, ps_train, grads)
            st_train = merge(st_train, (var_head=st_new.var_head, norm=st_new.norm, mixer=st_new.mixer, reasoner=st_new.reasoner))
            total_loss += loss_val
            n_updates += 1
            train_step += 1

            if i % 50 == 0
                @printf "Epoch %d [%d/%d] Loss: %.4f | Pred: %.4f | KL: %.4f | Recon: %.4f | Lcap: %d | |g|: %.4f | LR: %.2e\n" epoch i num_batches_per_epoch loss_val internals.pred internals.kl internals.recon internals.lcap grad_norm current_lr
            end

            if cfg.save_every > 0 && (i % cfg.save_every == 0)
                jldsave(
                    joinpath(cfg.checkpoint_dir, "$(cfg.checkpoint_prefix)_epoch$(epoch)_step$(i).jld2");
                    ps=ps_train |> cpu,
                    st=st_train |> cpu,
                    opt_state=opt_state |> cpu,
                    epoch=epoch,
                    step=i,
                    train_step=train_step,
                    stage1_ckpt=cfg.stage1_ckpt,
                    k_streams=cfg.k_streams,
                    heads=cfg.heads,
                    num_layers=cfg.num_layers,
                    block_size=cfg.block_size,
                    k_drop_threshold=cfg.k_drop_threshold,
                    k_drop_min=cfg.k_drop_min,
                )
            end
        end

        avg_loss = total_loss / max(n_updates, 1)
        println("Epoch $epoch Completed. Avg Loss: $avg_loss")
        flush(stdout)

        jldsave(
            joinpath(cfg.checkpoint_dir, "$(cfg.checkpoint_prefix)_epoch$epoch.jld2");
            ps=ps_train |> cpu,
            st=st_train |> cpu,
            opt_state=opt_state |> cpu,
            epoch=epoch,
            step=0,
            train_step=train_step,
            stage1_ckpt=cfg.stage1_ckpt,
            k_streams=cfg.k_streams,
            heads=cfg.heads,
            num_layers=cfg.num_layers,
            block_size=cfg.block_size,
            k_drop_threshold=cfg.k_drop_threshold,
            k_drop_min=cfg.k_drop_min,
        )
    end
end

function train_stage2(args::Vector{String})
    if "--help" in args || "-h" in args
        println("Usage: train_stage2 [options]")
        println("Options:")
        println("  --data-file <path>        Path to data file")
        println("  --meta-file <path>        Path to meta file")
        println("  --stage1-ckpt <path>      Stage1 checkpoint file")
        println("  --checkpoint-dir <path>   Directory for saving checkpoints")
        println("  --checkpoint-prefix <str> Prefix for checkpoint filenames")
        println("  --epochs <int>            Number of epochs (default: $EPOCHS)")
        println("  --batch-size <int>        Batch size (default: $BATCH_SIZE)")
        println("  --lr <float>              Learning rate (default: $LR)")
        println("  --max-batches <int>       Max batches per epoch (default: $MAX_BATCHES)")
        println("  --save-every <int>        Save every N steps (default: $SAVE_EVERY)")
        println("  --warmup-steps <int>      Warmup steps (default: $WARMUP_STEPS)")
        println("  --k-streams <int>         Number of streams K (default: $K_STREAMS)")
        println("  --heads <int>             Attention heads (default: $HEADS)")
        println("  --num-layers <int>        Number of MHC blocks (default: $NUM_LAYERS)")
        println("  --block-size <int>        Encoder block size (default: $BLOCK_SIZE)")
        println("  --seq-len <int>           Stream seq len (default: $SEQ_LEN)")
        println("  --grad-clip-norm <float>  Gradient clip norm (default: $GRAD_CLIP_NORM)")
        println("  --loss-spike-threshold <float> Loss spike threshold (default: $LOSS_SPIKE_THRESHOLD)")
        println("  --skip-on-spike           Skip updates on loss spikes (default: $SKIP_ON_SPIKE)")
        println("  --add-timestamp           Add timestamp to checkpoint prefix (default: $ADD_TIMESTAMP)")
        println("  --k-drop-threshold <float> K-drop threshold (default: $K_DROP_THRESHOLD)")
        println("  --k-drop-min <int>        Minimum kept streams after K-drop (default: $K_DROP_MIN)")
        println("  --force-cpu               Force CPU usage")
        return
    end
    cli = Utils.parse_cli_args(args)
    cfg = resolve_config(cli)
    train(cfg)
end

end
