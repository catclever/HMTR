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

    return (; data_file, meta_file, checkpoint_dir, checkpoint_prefix, stage1_ckpt, resume_ckpt, epochs, batch_size, lr, max_batches, save_every, warmup_steps, k_streams, heads, num_layers, block_size, seq_len, skip_on_spike, force_cpu, grad_clip_norm, loss_spike_threshold)
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
    if vocab_ckpt != vocab_size
        error("Vocab size mismatch: ckpt vocab_size=$(vocab_ckpt) but data vocab_size=$(vocab_size)")
    end
    mamba_d_state = if haskey(ckpt, "mamba_d_state")
        Int(ckpt["mamba_d_state"])
    else
        ds = infer_mamba_d_state_from_ps(ps, dim)
        ds === nothing ? 16 : Int(ds)
    end
    model = HMTR_Stage1_AutoEncoder(vocab_size, dim; block_size=cfg.block_size, pad_id=pad_id, eos_id=eos_id, mamba_d_state=mamba_d_state)
    rng = Random.default_rng()
    Random.seed!(rng, 42)
    ps0, st0 = Lux.setup(rng, model)
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

function encode_capsules(model, ps, st, x_batch)
    capsules_params, st_enc = model.encoder(x_batch, ps.encoder, st.encoder)
    mu, logvar = capsules_params
    z = Model.reparameterize(mu, logvar; training=false)
    z_norm, st_norm = model.norm(z, ps.norm, st.norm)
    st_new = merge(st, (encoder=st_enc, norm=st_norm))
    return z_norm, st_new
end

function compute_loss(mixer, reasoner, ps2, st2, capsules)
    x_k, st_mix = mixer(capsules, ps2.mixer, st2.mixer)
    y, st_reas = reasoner(x_k, ps2.reasoner, st2.reasoner)
    y_mean = dropdims(mean(y; dims=2); dims=2)
    Lcap = size(y_mean, 2)
    if Lcap <= 1
        loss = zero(eltype(y_mean))
        return loss, (mixer=st_mix, reasoner=st_reas), (; pred=loss, lcap=Lcap)
    end
    target = Zygote.dropgrad(@view capsules[:, 2:end, :])
    pred = @view y_mean[:, 1:end-1, :]
    loss = mean(abs2, pred .- target)
    return loss, (mixer=st_mix, reasoner=st_reas), (; pred=loss, lcap=Lcap)
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
    reasoner = MHCLatentReasoner(dim, cfg.k_streams, cfg.heads, cfg.num_layers)
    ps_mix, st_mix = Lux.setup(rng, mixer)
    ps_reas, st_reas = Lux.setup(rng, reasoner)
    ps2 = (; mixer=ps_mix, reasoner=ps_reas)
    st2 = (; mixer=st_mix, reasoner=st_reas)

    resume_opt_state = nothing
    train_step = 0
    if !isempty(cfg.resume_ckpt)
        if !isfile(cfg.resume_ckpt)
            error("resume_ckpt=$(cfg.resume_ckpt) not found")
        end
        ckpt = JLD2.load(cfg.resume_ckpt)
        ps2 = ckpt["ps"]
        st2 = ckpt["st"]
        if haskey(ckpt, "opt_state")
            resume_opt_state = ckpt["opt_state"]
        end
        if haskey(ckpt, "train_step")
            train_step = Int(ckpt["train_step"])
        end
        println("Resuming from ckpt=$(cfg.resume_ckpt) train_step=$(train_step)")
    end

    ps2 = ps2 |> dev
    st2 = st2 |> dev

    opt = Optimisers.Adam(cfg.lr)
    opt_state = Optimisers.setup(opt, ps2)
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

            TrainStage1.apply_reset!(st1, reset_mask)
            TrainStage1.apply_reset!(st2, reset_mask)

            x_batch = x_cpu_raw |> dev
            capsules_norm, st1_new = encode_capsules(model1, ps1, st1, x_batch)
            st1 = st1_new
            capsules_norm = Zygote.dropgrad(capsules_norm)

            (loss, st2_new, internals), back = Zygote.pullback(
                p -> compute_loss(mixer, reasoner, p, st2, capsules_norm), ps2
            )

            loss_val = Float32(loss)
            spike = !isfinite(loss_val) || loss_val > Float32(cfg.loss_spike_threshold)
            if spike
                stats = TrainStage1.batch_stats(x_batch, vocab_size, pad_id, eos_id)
                pad_frac = stats.n_pad / max(stats.n_total, 1)
                eos_frac = stats.n_eos / max(stats.n_total, 1)
                @printf "SPIKE Epoch %d Step %d Loss %.4f | x[min=%d max=%d] pad=%.3f eos=%.3f\n" epoch i loss_val stats.x_min stats.x_max pad_frac eos_frac
                if !isfinite(loss_val)
                    st2 = st2_new
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

            opt_state, ps2 = Optimisers.update(opt_state, ps2, grads)
            st2 = st2_new
            total_loss += loss_val
            n_updates += 1
            train_step += 1

            if i % 50 == 0
                @printf "Epoch %d [%d/%d] Loss: %.4f | Pred: %.4f | Lcap: %d | |g|: %.4f | LR: %.2e\n" epoch i num_batches_per_epoch loss_val internals.pred internals.lcap grad_norm current_lr
            end

            if cfg.save_every > 0 && (i % cfg.save_every == 0)
                jldsave(
                    joinpath(cfg.checkpoint_dir, "$(cfg.checkpoint_prefix)_epoch$(epoch)_step$(i).jld2");
                    ps=ps2 |> cpu,
                    st=st2 |> cpu,
                    opt_state=opt_state |> cpu,
                    epoch=epoch,
                    step=i,
                    train_step=train_step,
                    stage1_ckpt=cfg.stage1_ckpt,
                    k_streams=cfg.k_streams,
                    heads=cfg.heads,
                    num_layers=cfg.num_layers,
                    block_size=cfg.block_size,
                )
            end
        end

        avg_loss = total_loss / max(n_updates, 1)
        println("Epoch $epoch Completed. Avg Loss: $avg_loss")
        flush(stdout)

        jldsave(
            joinpath(cfg.checkpoint_dir, "$(cfg.checkpoint_prefix)_epoch$epoch.jld2");
            ps=ps2 |> cpu,
            st=st2 |> cpu,
            opt_state=opt_state |> cpu,
            epoch=epoch,
            step=0,
            train_step=train_step,
            stage1_ckpt=cfg.stage1_ckpt,
            k_streams=cfg.k_streams,
            heads=cfg.heads,
            num_layers=cfg.num_layers,
            block_size=cfg.block_size,
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
        println("  --force-cpu               Force CPU usage")
        return
    end
    cli = Utils.parse_cli_args(args)
    cfg = resolve_config(cli)
    train(cfg)
end

end
