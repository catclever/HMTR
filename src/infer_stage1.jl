module InferStage1

using Lux
using Random
using JLD2
using Printf
using CUDA
using LuxCUDA
using Statistics
using ..Utils
using ..Model

export infer_stage1, infer_stage2

const PROJECT_ROOT = normpath(joinpath(@__DIR__, ".."))
const DATA_DIR = get(ENV, "DATA_DIR", joinpath(PROJECT_ROOT, "data"))
const DATA_FILE = get(ENV, "DATA_FILE", joinpath(DATA_DIR, "processed.jld2"))
const META_FILE = get(ENV, "META_FILE", "")
const CHECKPOINT_DIR = get(ENV, "CHECKPOINT_DIR", joinpath(PROJECT_ROOT, "checkpoints"))
const CHECKPOINT_FILE = get(ENV, "CHECKPOINT_FILE", "")
const STAGE1_CKPT = get(ENV, "STAGE1_CKPT", "")
const TEXT = get(ENV, "TEXT", "")
const SHOW_IDS = parse(Int, get(ENV, "SHOW_IDS", "0"))
const FORCE_CPU = parse(Int, get(ENV, "FORCE_CPU", "0"))
const INTERACTIVE = parse(Int, get(ENV, "INTERACTIVE", "0"))
const SAMPLE = parse(Int, get(ENV, "SAMPLE", "0"))
const PREDICT = parse(Int, get(ENV, "PREDICT", "0"))

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
    elseif !isabspath(meta_file)
        if !isfile(meta_file) && isfile(joinpath(DATA_DIR, meta_file))
            meta_file = joinpath(DATA_DIR, meta_file)
        end
    end

    checkpoint_file = get(cli, :checkpoint_file, CHECKPOINT_FILE)
    if isempty(checkpoint_file)
        error("--checkpoint-file is required (or set CHECKPOINT_FILE env)")
    end

    checkpoint_dir = get(cli, :checkpoint_dir, CHECKPOINT_DIR)
    # User request: Do not auto-prepend checkpoint_dir. Look in CWD if relative.
    # if !isabspath(checkpoint_file)
    #     checkpoint_file = joinpath(checkpoint_dir, checkpoint_file)
    # end
    if !endswith(checkpoint_file, ".jld2") && isfile(checkpoint_file * ".jld2")
        checkpoint_file = checkpoint_file * ".jld2"
    end

    text = string(get(cli, :text, TEXT))

    show_ids_raw = string(get(cli, :show_ids, SHOW_IDS))
    force_cpu_raw = string(get(cli, :force_cpu, FORCE_CPU))
    interactive_raw = string(get(cli, :interactive, INTERACTIVE))
    sample_raw = string(get(cli, :sample, SAMPLE))
    predict_raw = string(get(cli, :predict, PREDICT))
    show_ids = show_ids_raw == "true" || show_ids_raw == "1"
    force_cpu = force_cpu_raw == "true" || force_cpu_raw == "1"
    interactive = interactive_raw == "true" || interactive_raw == "1"
    sample = sample_raw == "true" || sample_raw == "1"
    predict = predict_raw == "true" || predict_raw == "1"

    if isempty(text) && !interactive
        error("--text is required unless --interactive is enabled")
    end

    return (; data_file, meta_file, checkpoint_file, text, show_ids, force_cpu, interactive, sample, predict)
end

function resolve_config_stage2(cli::Dict{Symbol, Any})
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
    elseif !isabspath(meta_file)
        if !isfile(meta_file) && isfile(joinpath(DATA_DIR, meta_file))
            meta_file = joinpath(DATA_DIR, meta_file)
        end
    end

    checkpoint_file = get(cli, :checkpoint_file, CHECKPOINT_FILE)
    if isempty(checkpoint_file)
        error("--checkpoint-file is required (or set CHECKPOINT_FILE env)")
    end
    if !endswith(checkpoint_file, ".jld2") && isfile(checkpoint_file * ".jld2")
        checkpoint_file = checkpoint_file * ".jld2"
    end

    stage1_ckpt = string(get(cli, :stage1_ckpt, get(cli, :stage1_checkpoint, STAGE1_CKPT)))

    text = string(get(cli, :text, TEXT))

    show_ids_raw = string(get(cli, :show_ids, SHOW_IDS))
    force_cpu_raw = string(get(cli, :force_cpu, FORCE_CPU))
    interactive_raw = string(get(cli, :interactive, INTERACTIVE))
    sample_raw = string(get(cli, :sample, SAMPLE))
    show_ids = show_ids_raw == "true" || show_ids_raw == "1"
    force_cpu = force_cpu_raw == "true" || force_cpu_raw == "1"
    interactive = interactive_raw == "true" || interactive_raw == "1"
    sample = sample_raw == "true" || sample_raw == "1"

    if isempty(text) && !interactive
        error("--text is required unless --interactive is enabled")
    end

    return (; data_file, meta_file, checkpoint_file, stage1_ckpt, text, show_ids, force_cpu, interactive, sample)
end

function select_device(force_cpu::Bool)
    if force_cpu
        return cpu_device()
    end
    return CUDA.functional() ? gpu_device() : cpu_device()
end

function load_meta(data_file::AbstractString, meta_file::AbstractString)
    if isfile(meta_file)
        data_file = meta_file
    elseif !isfile(data_file)
        error("Meta file $(meta_file) not found and data file $(data_file) not found!")
    end
    params = nothing
    char_map = Dict{Any, Any}()
    vocab = nothing

    function load_key(f, key, default)
        haskey(f, key) || return default
        v = f[key]
        try
            return read(v)
        catch
            return v
        end
    end

    JLD2.jldopen(data_file, "r") do f
        params = load_key(f, "params", nothing)
        char_map = load_key(f, "char_map", Dict{Any, Any}())
        vocab = load_key(f, "vocab", nothing)
    end
    params === nothing && error("data file missing key \"params\"")

    block_size = get(params, "BLOCK_SIZE", 8)
    pad_id = params["PAD"]
    eos_id = params["EOS"]
    unk_id = get(params, "UNK", 0)

    vocab_size = if haskey(params, "VOCAB_SIZE")
        params["VOCAB_SIZE"]
    elseif !isempty(char_map)
        length(char_map) + 3
    elseif vocab !== nothing
        length(vocab) - 1
    else
        error("Cannot infer vocab_size from data file")
    end

    char_map2 = Dict{Char, Int}()
    if !isempty(char_map)
        for (k, v) in char_map
            if k isa Char
                char_map2[k] = Int(v)
            elseif k isa AbstractString && ncodeunits(k) > 0
                char_map2[first(k)] = Int(v)
            end
        end
    end

    id_to_char = Dict{Int, Char}()
    if !isempty(char_map2)
        for (c, tid) in char_map2
            id_to_char[Int(tid)] = c
        end
    end

    return (; block_size, vocab_size, pad_id, eos_id, unk_id, char_map=char_map2, id_to_char, vocab)
end

normalize_text(x) = x === missing ? "" : String(x)

function encode_text_to_blocks(text::AbstractString, meta)
    s = normalize_text(text)
    ids = Vector{Int}()
    if !isempty(meta.char_map)
        for c in s
            push!(ids, get(meta.char_map, c, meta.unk_id))
        end
    else
        error("char_map is empty; this inference mode expects char-tokenized data")
    end
    push!(ids, meta.eos_id)

    K = meta.block_size
    n_blocks = div(length(ids) + K - 1, K)
    x = Matrix{Int}(undef, K, n_blocks)
    pos = 1
    for b in 1:n_blocks
        @inbounds for j in 1:K
            src = pos + j - 1
            x[j, b] = src <= length(ids) ? ids[src] : meta.pad_id
        end
        pos += K
    end
    return x
end

function encode_text_to_seq(text::AbstractString, meta)
    s = normalize_text(text)
    ids = Vector{Int}()
    if !isempty(meta.char_map)
        for c in s
            push!(ids, get(meta.char_map, c, meta.unk_id))
        end
    else
        error("char_map is empty; this inference mode expects char-tokenized data")
    end
    push!(ids, meta.eos_id)
    return reshape(ids, :, 1)
end

function decode_ids(ids::AbstractVector{Int}, meta)
    out = IOBuffer()
    for tid in ids
        if tid == meta.eos_id
            break
        elseif tid == meta.pad_id
            continue
        elseif tid == meta.unk_id
            write(out, '?')
        elseif haskey(meta.id_to_char, tid)
            write(out, meta.id_to_char[tid])
        else
            # write(out, "")
        end
    end
    return String(take!(out))
end

function decode_blocks(x::AbstractMatrix{Int}, meta)
    out = IOBuffer()
    for b in 1:size(x, 2)
        s = decode_ids(vec(@view x[:, b]), meta)
        write(out, s)
        if occursin('\0', s)
            break
        end
    end
    return String(take!(out))
end

function logits_to_ids(logits)
    V, L, B = size(logits)
    ids = Matrix{Int}(undef, L, B)
    for b in 1:B
        for t in 1:L
            col = @view logits[:, t, b]
            _, idx = findmax(col)
            ids[t, b] = Int(idx)
        end
    end
    return ids
end

function stop_after_eos(ids::AbstractMatrix{Int}, eos_id::Int)
    L, B = size(ids)
    out = copy(ids)
    for b in 1:B
        pos = findfirst(==(eos_id), @view out[:, b])
        if pos !== nothing && pos < L
            out[pos + 1:end, b] .= eos_id
        end
    end
    return out
end

function greedy_decode_capsules(decoder, capsules, ps_dec, st_dec, meta, dev, cpu)
    D, L_cap, B = size(capsules)
    K = hasproperty(decoder, :block_size) ? getfield(decoder, :block_size) : meta.block_size

    latents_flat = copy(reshape(capsules, D, :))
    N_capsules = size(latents_flat, 2)

    h = latents_flat
    vocab_size = size(ps_dec.proj.weight, 1)
    out_buf = similar(latents_flat, vocab_size, K, N_capsules)

    st_emb = st_dec.embedding
    st_cell = st_dec.cell
    st_proj = st_dec.proj

    prev_ids = fill(Int(meta.eos_id), N_capsules)
    x_in, st_emb = decoder.embedding(prev_ids |> dev, ps_dec.embedding, st_emb)

    for k in 1:K
        (out, (h_new,)), st_cell = decoder.cell((x_in, (h,)), ps_dec.cell, st_cell)
        h = h_new
        logits, st_proj = decoder.proj(out, ps_dec.proj, st_proj)
        out_buf[:, k, :] = logits

        if k < K
            logits_cpu = logits |> cpu
            prev_ids = Vector{Int}(undef, size(logits_cpu, 2))
            for i in 1:size(logits_cpu, 2)
                _, idx = findmax(@view logits_cpu[:, i])
                prev_ids[i] = Int(idx)
            end
            x_in, st_emb = decoder.embedding(prev_ids |> dev, ps_dec.embedding, st_emb)
        end
    end

    out_reshaped = reshape(out_buf, size(out_buf, 1), K, L_cap, B)
    final_logits = reshape(out_reshaped, size(out_reshaped, 1), :, B)
    return logits_to_ids(final_logits |> cpu)
end

function infer_once(text::AbstractString, model, ps, st, meta, dev, cpu, rng; show_ids::Bool, sample::Bool, predict::Bool)
    x = encode_text_to_seq(text, meta)
    x_dev = x |> dev

    capsules, _st_enc = model.encoder(x_dev, ps.encoder, st.encoder)
    if capsules isa Tuple
        mu, logvar = capsules
        capsules = sample ? Model.reparameterize(mu, logvar; rng=rng, training=true) : mu
    end
    capsules_norm, _st_norm = model.norm(capsules, ps.norm, st.norm)
    pred = greedy_decode_capsules(model.decoder, capsules_norm, ps.decoder, st.decoder, meta, dev, cpu)
    pred = stop_after_eos(pred, meta.eos_id)

    input_text = decode_ids(vec(x), meta)
    pred_text = decode_ids(vec(pred), meta)

    println("Input:  ", text)
    println("Tokens: ", input_text)
    println("Recons: ", pred_text)
    if predict
        z_pred, _st_pred = model.predictor(capsules_norm, ps.predictor, st.predictor)
        if size(z_pred, 2) > 1
            z_next = @view z_pred[:, 1:end-1, :]
            pred_next = greedy_decode_capsules(model.decoder, z_next, ps.decoder, st.decoder, meta, dev, cpu)
            pred_next = stop_after_eos(pred_next, meta.eos_id)
            pred_next_text = decode_ids(vec(pred_next), meta)
            println("PredNext: ", pred_next_text)
        else
            println("PredNext: ")
        end
    end

    if show_ids
        println("InputIds:")
        show(stdout, "text/plain", x)
        println()
        println("PredIds:")
        show(stdout, "text/plain", pred)
        println()
    end
    return
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

function load_stage1_decoder(meta, stage1_ckpt::AbstractString)
    if isempty(stage1_ckpt) || !isfile(stage1_ckpt)
        error("stage1_ckpt=$(stage1_ckpt) not found")
    end
    ckpt = JLD2.load(stage1_ckpt)
    ps = ckpt["ps"]
    st = ckpt["st"]

    dim = size(ps.encoder.embedding.weight, 1)
    vocab_ckpt = size(ps.encoder.embedding.weight, 2)
    vocab_size = meta.vocab_size

    mamba_d_state = if haskey(ckpt, "mamba_d_state")
        Int(ckpt["mamba_d_state"])
    else
        ds = infer_mamba_d_state_from_ps(ps, dim)
        ds === nothing ? 16 : Int(ds)
    end

    model = HMTR_Stage1_AutoEncoder(vocab_size, dim; block_size=meta.block_size, pad_id=meta.pad_id, eos_id=meta.eos_id, mamba_d_state=mamba_d_state)
    rng = Random.default_rng()
    Random.seed!(rng, 42)
    ps0, st0 = Lux.setup(rng, model)

    if vocab_ckpt != vocab_size
        n_copy = min(vocab_ckpt, vocab_size)
        new_enc_embed = copy(ps0.encoder.embedding.weight)
        new_enc_embed[:, 1:n_copy] .= ps.encoder.embedding.weight[:, 1:n_copy]
        ps = merge(ps, (encoder=merge(ps.encoder, (embedding=(weight=new_enc_embed,),)),))

        if haskey(ps, :decoder)
            new_dec_embed = copy(ps0.decoder.embedding.weight)
            if haskey(ps.decoder, :embedding)
                new_dec_embed[:, 1:n_copy] .= ps.decoder.embedding.weight[:, 1:n_copy]
            end

            new_dec_proj_w = copy(ps0.decoder.proj.weight)
            new_dec_proj_b = copy(ps0.decoder.proj.bias)
            if haskey(ps.decoder, :proj)
                new_dec_proj_w[1:n_copy, :] .= ps.decoder.proj.weight[1:n_copy, :]
                if length(size(ps.decoder.proj.bias)) >= 2
                    new_dec_proj_b[1:n_copy, 1] .= ps.decoder.proj.bias[1:n_copy, 1]
                else
                    new_dec_proj_b[1:n_copy] .= ps.decoder.proj.bias[1:n_copy]
                end
            end
            ps = merge(ps, (decoder=merge(ps.decoder, (embedding=(weight=new_dec_embed,), proj=(weight=new_dec_proj_w, bias=new_dec_proj_b))),))
        end
    end

    if !haskey(ps, :norm) || !haskey(st, :norm)
        if !haskey(ps, :norm)
            ps = merge(ps, (norm=ps0.norm,))
        end
        if !haskey(st, :norm)
            st = merge(st, (norm=st0.norm,))
        end
    end

    return model, ps, st
end

function infer(cfg)
    meta = load_meta(cfg.data_file, cfg.meta_file)

    ckpt = JLD2.load(cfg.checkpoint_file)
    ps = ckpt["ps"]
    st = ckpt["st"]

    dim = size(ps.encoder.embedding.weight, 1)
    vocab_size = size(ps.encoder.embedding.weight, 2)
    if haskey(ps.encoder, :head) && (!haskey(ps.encoder, :mu_head) || !haskey(ps.encoder, :var_head))
        stage1_head = ps.encoder.head
        mu_w = stage1_head.weight[1:dim, :]
        var_w = stage1_head.weight[dim+1:end, :]
        mu_b = stage1_head.bias[1:dim]
        var_b = stage1_head.bias[dim+1:end]
        encoder_full = (embedding=ps.encoder.embedding, layers=ps.encoder.layers, mu_head=(weight=mu_w, bias=mu_b), var_head=(weight=var_w, bias=var_b))
        ps = merge(ps, (encoder=encoder_full,))
    end
    if haskey(st.encoder, :head) && (!haskey(st.encoder, :mu_head) || !haskey(st.encoder, :var_head))
        encoder_st = (embedding=st.encoder.embedding, layers=st.encoder.layers, mu_head=st.encoder.head, var_head=st.encoder.head)
        st = merge(st, (encoder=encoder_st,))
    end
    if vocab_size != meta.vocab_size
        error("Vocab size mismatch: ckpt vocab_size=$(vocab_size) but data meta vocab_size=$(meta.vocab_size)")
    end

    mamba_d_state = if haskey(ckpt, "mamba_d_state")
        Int(ckpt["mamba_d_state"])
    else
        ds = infer_mamba_d_state_from_ps(ps, dim)
        ds === nothing ? 16 : Int(ds)
    end

    model = HMTR_Stage1_AutoEncoder(vocab_size, dim; block_size=meta.block_size, pad_id=meta.pad_id, eos_id=meta.eos_id, mamba_d_state=mamba_d_state)

    dev = select_device(cfg.force_cpu)
    cpu = cpu_device()

    if !haskey(ps, :norm) || !haskey(st, :norm)
        rng = Random.default_rng()
        Random.seed!(rng, 42)
        ps0, st0 = Lux.setup(rng, model)
        if !haskey(ps, :norm)
            ps = merge(ps, (norm=ps0.norm,))
        end
        if !haskey(st, :norm)
            st = merge(st, (norm=st0.norm,))
        end
    end

    ps = ps |> dev
    st = st |> dev

    rng = Random.default_rng()

    if cfg.interactive
        while true
            print("> ")
            line = readline(stdin; keep=true)
            s = strip(line)
            if isempty(s) || s == ":q" || s == "quit" || s == "exit"
                break
            end
            infer_once(s, model, ps, st, meta, dev, cpu, rng; show_ids=cfg.show_ids, sample=cfg.sample, predict=cfg.predict)
        end
        return
    end

    infer_once(cfg.text, model, ps, st, meta, dev, cpu, rng; show_ids=cfg.show_ids, sample=cfg.sample, predict=cfg.predict)
    return
end

function infer_stage2_once(text::AbstractString, meta, ps2, st2, decoder_model, ps_dec, st_dec, mixer, reasoner, k_streams::Int, dev, cpu, rng; show_ids::Bool, sample::Bool)
    x = encode_text_to_seq(text, meta)
    x_dev = x |> dev

    capsules_params, st_enc = decoder_model.encoder(x_dev, ps2.encoder, st2.encoder)
    if capsules_params isa Tuple
        mu, logvar = capsules_params
        capsules = sample ? Model.reparameterize(mu, logvar; rng=rng, training=true) : mu
    else
        capsules = capsules_params
    end

    capsules_norm, _st_norm = decoder_model.norm(capsules, ps2.norm, st2.norm)
    pred = greedy_decode_capsules(decoder_model.decoder, capsules_norm, ps_dec.decoder, st_dec.decoder, meta, dev, cpu)
    pred = stop_after_eos(pred, meta.eos_id)

    input_text = decode_ids(vec(x), meta)
    pred_text = decode_ids(vec(pred), meta)

    println("Input:  ", text)
    println("Tokens: ", input_text)
    println("Recons: ", pred_text)

    x_k_unnorm, _st_mix = mixer(capsules_params, k_streams, ps2.mixer, st2.mixer)
    x_k_flat = reshape(x_k_unnorm, size(capsules, 1), :)
    x_k_norm_flat, _st_norm2 = decoder_model.norm(x_k_flat, ps2.norm, st2.norm)
    x_k = reshape(x_k_norm_flat, size(capsules, 1), k_streams, size(x_k_unnorm, 3), size(x_k_unnorm, 4))
    y, _st_reas = reasoner(x_k, ps2.reasoner, st2.reasoner)
    y_mean = dropdims(mean(y; dims=2); dims=2)

    if size(y_mean, 2) > 1
        z_next = @view y_mean[:, 1:end-1, :]
        pred_next = greedy_decode_capsules(decoder_model.decoder, z_next, ps_dec.decoder, st_dec.decoder, meta, dev, cpu)
        pred_next = stop_after_eos(pred_next, meta.eos_id)
        pred_next_text = decode_ids(vec(pred_next), meta)
        println("PredNext: ", pred_next_text)
    else
        println("PredNext: ")
    end

    if show_ids
        println("InputIds:")
        show(stdout, "text/plain", x)
        println()
        println("PredIds:")
        show(stdout, "text/plain", pred)
        println()
    end
    return
end

function infer_stage2(args::Vector{String})
    if "--help" in args || "-h" in args
        println("Usage: infer_stage2 [options]")
        println("Options:")
        println("  --checkpoint-file <path>  Stage2 checkpoint file (REQUIRED)")
        println("  --stage1-ckpt <path>       Stage1 checkpoint file (for decoder)")
        println("  --text <string>            Text to reconstruct")
        println("  --interactive              Interactive mode")
        println("  --data-file <path>         Path to data file (for vocab/meta)")
        println("  --show-ids                 Show token IDs")
        println("  --force-cpu                Force CPU usage")
        println("  --sample                   Enable VAE resampling")
        return
    end
    cli = Utils.parse_cli_args(args)
    cfg = resolve_config_stage2(cli)

    meta = load_meta(cfg.data_file, cfg.meta_file)

    ckpt = JLD2.load(cfg.checkpoint_file)
    ps2 = ckpt["ps"]
    st2 = ckpt["st"]

    stage1_ckpt = cfg.stage1_ckpt
    if isempty(stage1_ckpt) && haskey(ckpt, "stage1_ckpt")
        stage1_ckpt = string(ckpt["stage1_ckpt"])
    end
    decoder_model, ps_dec, st_dec = load_stage1_decoder(meta, stage1_ckpt)

    ps2_full = ps2
    st2_full = st2
    if !haskey(ps2, :encoder)
        stage1_enc = ps_dec.encoder
        encoder_full = if haskey(stage1_enc, :mu_head) && haskey(stage1_enc, :var_head)
            (embedding=stage1_enc.embedding, layers=stage1_enc.layers, mu_head=stage1_enc.mu_head, var_head=ps2.var_head)
        else
            stage1_head = stage1_enc.head
            dim_stage1 = size(stage1_enc.embedding.weight, 1)
            mu_w = stage1_head.weight[1:dim_stage1, :]
            var_w = stage1_head.weight[dim_stage1+1:end, :]
            mu_b = stage1_head.bias[1:dim_stage1]
            var_b = stage1_head.bias[dim_stage1+1:end]
            (embedding=stage1_enc.embedding, layers=stage1_enc.layers, mu_head=(weight=mu_w, bias=mu_b), var_head=(weight=var_w, bias=var_b))
        end
        ps2_full = merge(ps2, (encoder=encoder_full, norm=ps2.norm))
    end
    if !haskey(st2_full, :encoder)
        st2_full = merge(st2_full, (encoder=st_dec.encoder,))
    end
    if !haskey(st2_full, :norm)
        st2_full = merge(st2_full, (norm=st_dec.norm,))
    end

    dim = size(ps2_full.encoder.embedding.weight, 1)
    vocab_size = size(ps2_full.encoder.embedding.weight, 2)
    if vocab_size != meta.vocab_size
        error("Vocab size mismatch: stage2 vocab_size=$(vocab_size) but data meta vocab_size=$(meta.vocab_size)")
    end

    heads = haskey(ckpt, "heads") ? Int(ckpt["heads"]) : 8
    num_layers = haskey(ckpt, "num_layers") ? Int(ckpt["num_layers"]) : 4
    k_streams = haskey(ckpt, "k_streams") ? Int(ckpt["k_streams"]) : size(ps2.mixer.offsets, 2)
    k_drop_threshold = haskey(ckpt, "k_drop_threshold") ? Float32(ckpt["k_drop_threshold"]) : 0f0
    k_drop_min = haskey(ckpt, "k_drop_min") ? Int(ckpt["k_drop_min"]) : 1
    mixer = InitialMixingLayer(dim, k_streams)
    reasoner = MHCLatentReasoner(dim, k_streams, heads, num_layers; k_drop_threshold=k_drop_threshold, k_drop_min=k_drop_min)

    dev = select_device(cfg.force_cpu)
    cpu = cpu_device()

    moved = false
    if dev !== cpu
        try
            ps2 = ps2 |> dev
            st2 = st2 |> dev
            ps_dec = ps_dec |> dev
            st_dec = st_dec |> dev
            moved = true
        catch
            dev = cpu
        end
    end
    if !moved
        ps2 = ps2 |> dev
        st2 = st2 |> dev
        ps_dec = ps_dec |> dev
        st_dec = st_dec |> dev
    end

    rng = Random.default_rng()

    if cfg.interactive
        while true
            print("> ")
            line = readline(stdin; keep=true)
            s = strip(line)
            if isempty(s) || s == ":q" || s == "quit" || s == "exit"
                break
            end
            infer_stage2_once(s, meta, ps2_full, st2_full, decoder_model, ps_dec, st_dec, mixer, reasoner, k_streams, dev, cpu, rng; show_ids=cfg.show_ids, sample=cfg.sample)
        end
        return
    end

    infer_stage2_once(cfg.text, meta, ps2_full, st2_full, decoder_model, ps_dec, st_dec, mixer, reasoner, k_streams, dev, cpu, rng; show_ids=cfg.show_ids, sample=cfg.sample)
    return
end

function infer_stage1(args::Vector{String})
    if "--help" in args || "-h" in args
        println("Usage: infer_stage1 [options]")
        println("Options:")
        println("  --checkpoint-file <path>  Path to checkpoint file (REQUIRED)")
        println("  --text <string>           Text to reconstruct")
        println("  --interactive             Interactive mode")
        println("  --data-file <path>        Path to data file (for vocab/meta)")
        println("  --show-ids                Show token IDs")
        println("  --force-cpu               Force CPU usage")
        println("  --sample                  Enable VAE resampling")
        println("  --predict                 Decode predicted next capsule blocks")
        return
    end
    cli = Utils.parse_cli_args(args)
    cfg = resolve_config(cli)
    infer(cfg)
end

end # module Infer
