module Data

using Parquet2
using DataFrames
using JLD2
using Random
using PythonCall
using Dates
using ..Utils

export data_prep

const PROJECT_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const DEFAULT_DATA_DIR = get(ENV, "DATA_DIR", joinpath(PROJECT_ROOT, "data"))
const DEFAULT_PARQUET_FILE = get(ENV, "PARQUET_FILE", "")
const DEFAULT_TOKENIZER_NAME = get(ENV, "TOKENIZER_NAME", "")
const DEFAULT_MAX_DOCS = parse(Int, get(ENV, "MAX_DOCS", "0"))
const DEFAULT_CHAR_VOCAB_DOCS = parse(Int, get(ENV, "CHAR_VOCAB_DOCS", "10000"))
const DEFAULT_META_FILE = get(ENV, "META_FILE", "")

function resolve_config(cli::Dict{Symbol, Any})
    data_dir = string(get(cli, :data_dir, DEFAULT_DATA_DIR))
    parquet_file = string(get(cli, :parquet_file, DEFAULT_PARQUET_FILE))
    tokenizer_name = string(get(cli, :tokenizer_name, DEFAULT_TOKENIZER_NAME))
    max_docs = parse(Int, string(get(cli, :max_docs, DEFAULT_MAX_DOCS)))
    char_vocab_docs = parse(Int, string(get(cli, :char_vocab_docs, DEFAULT_CHAR_VOCAB_DOCS)))
    output_file = string(get(cli, :output_file, ""))
    meta_file = string(get(cli, :meta_file, DEFAULT_META_FILE))

    block_size_cli = parse(Int, string(get(cli, :block_size, "0")))
    char_bs_cli = parse(Int, string(get(cli, :char_block_size, get(cli, :char_bs, "0"))))
    tok_bs_cli = parse(Int, string(get(cli, :tokenizer_block_size, get(cli, :tok_bs, "0"))))

    block_size = if block_size_cli > 0
        block_size_cli
    elseif isempty(strip(tokenizer_name))
        char_bs_cli > 0 ? char_bs_cli : 32
    else
        tok_bs_cli > 0 ? tok_bs_cli : 8
    end

    if tok_bs_cli > 0 && isempty(strip(tokenizer_name))
        error("--tokenizer-name is required when using --tokenizer-block-size/--tok-bs")
    end

    return (; data_dir, parquet_file, tokenizer_name, max_docs, char_vocab_docs, block_size, output_file, meta_file)
end

function resolve_output_file(mode::AbstractString, data_dir::AbstractString, block_size::Int, tokenizer_name::AbstractString; output_file::AbstractString = "")
    if !isempty(output_file)
        return output_file
    end
    ts = Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
    if mode != "char" && isempty(strip(tokenizer_name))
        error("--tokenizer-name is required for tokenizer mode")
    end
    label = mode == "char" ? "char" : "tok_" * sanitize_component(tokenizer_name)
    return joinpath(data_dir, "processed_$(label)_bs$(block_size)_$(ts).jld2")
end

function resolve_meta_file(output_file::AbstractString; meta_file::AbstractString = "")
    if !isempty(meta_file)
        return meta_file
    end
    if endswith(output_file, ".jld2")
        base = output_file[1:end - 5]
        return base * "_meta.jld2"
    end
    return output_file * "_meta.jld2"
end

function save_metadata(meta_file::AbstractString; params, char_map, vocab)
    dir = dirname(meta_file)
    if !isempty(dir) && dir != "." && !isdir(dir)
        mkpath(dir)
    end
    jldsave(meta_file; params=params, char_map=char_map, vocab=vocab)
    return
end

function load_parquet_data(path::String)
    println("Loading Parquet file: $path")
    ds = Parquet2.Dataset(path)
    df = DataFrame(ds)
    
    col_names = names(df)
    println("Columns: $col_names")
    
    if "text" in col_names
        return df.text
    elseif "content" in col_names
        return df.content
    else
        return df[:, 1]
    end
end

normalize_text(x) = x === missing ? "" : String(x)

function init_tokenizer(tokenizer_name::AbstractString)
    transformers = try
        pyimport("transformers")
    catch
        sys = pyimport("sys")
        subprocess = pyimport("subprocess")
        try
            subprocess.check_call([sys.executable, "-m", "ensurepip", "--upgrade"])
        catch
        end
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "transformers", "tokenizers"])
        pyimport("transformers")
    end
    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name)

    function try_pyint(x)
        try
            return pyconvert(Int, x)
        catch
            return nothing
        end
    end

    pad_id = something(try_pyint(tokenizer.pad_token_id), 0)
    eos_id = something(try_pyint(tokenizer.eos_token_id), try_pyint(tokenizer.sep_token_id), try_pyint(tokenizer.cls_token_id), pad_id)
    unk_id = something(try_pyint(tokenizer.unk_token_id), -1)
    vocab_size = something(try_pyint(tokenizer.vocab_size), try_pyint(length(tokenizer)), 0)

    punct_tokens = ["。", "！", "？", "；", ".", "!", "?", ";"]
    punct_ids = Set{Int}()
    for tok in punct_tokens
        tid = try_pyint(tokenizer.convert_tokens_to_ids(tok))
        if isnothing(tid)
            continue
        end
        if tid != unk_id
            push!(punct_ids, tid)
        end
    end

    vocab_py = tokenizer.get_vocab()
    vocab_jl = pyconvert(Dict{String, Int}, vocab_py)
    max_id = maximum(values(vocab_jl))
    id_to_token = fill("", max(vocab_size, max_id + 1))
    for (tok, tid) in vocab_jl
        id_to_token[tid + 1] = tok
    end

    return tokenizer, vocab_size, pad_id, eos_id, unk_id, punct_ids, id_to_token
end

function encode_ids(tokenizer, s::String)
    ids_py = tokenizer.encode(s; add_special_tokens=false)
    return pyconvert(Vector{Int}, ids_py)
end

function build_char_set(texts)
    chars = Set{Char}()
    for t in texts
        s = normalize_text(t)
        for c in s
            push!(chars, c)
        end
    end
    return chars
end

function split_sentences_zh(s::AbstractString)
    delims = Set(['。', '；', '？', '！'])
    out = String[]
    buf = IOBuffer()
    for c in s
        write(buf, c)
        if c in delims
            seg = strip(String(take!(buf)))
            isempty(seg) || push!(out, seg)
        end
    end
    tail = strip(String(take!(buf)))
    isempty(tail) || push!(out, tail)
    return out
end

function count_total_blocks_char(texts, block_size::Int)
    total = 0
    for doc in texts
        s = normalize_text(doc)
        isempty(s) && continue
        total += length(split_sentences_zh(s))
    end
    return total
end

function encode_chars(s::String, char_map, unk_id::Int)
    out = Vector{Int}(undef, length(s))
    i = 1
    for c in s
        out[i] = get(char_map, c, unk_id)
        i += 1
    end
    return out
end

function process_data_char(texts, output_file::AbstractString, meta_file::AbstractString, block_size::Int, char_vocab_docs::Int)
    sample_texts = texts[1:min(end, char_vocab_docs)]
    vocab_chars = sort!(collect(build_char_set(sample_texts)))

    char_map = Dict{Char, Int}(c => i + 3 for (i, c) in enumerate(vocab_chars))
    PAD_ID = 1
    EOS_ID = 2
    UNK_ID = 3
    vocab_size = length(char_map) + 3

    total_blocks = count_total_blocks_char(texts, block_size)
    println("Vocabulary size: $vocab_size")
    println("Total blocks to generate: $total_blocks")

    data_matrix = Matrix{Int}(undef, block_size, total_blocks)
    col = 1
    for (idx, doc) in enumerate(texts)
        if idx % 1000 == 0
            println("Processing doc $idx / $(length(texts))")
        end

        s = normalize_text(doc)
        isempty(s) && continue
        for seg in split_sentences_zh(s)
            ids = encode_chars(seg, char_map, UNK_ID)
            if length(ids) >= block_size
                ids = ids[1:block_size]
                @inbounds for j in 1:block_size
                    data_matrix[j, col] = ids[j]
                end
            else
                @inbounds begin
                    for j in 1:length(ids)
                        data_matrix[j, col] = ids[j]
                    end
                    data_matrix[length(ids) + 1, col] = EOS_ID
                    for j in (length(ids) + 2):block_size
                        data_matrix[j, col] = PAD_ID
                    end
                end
            end
            col += 1
        end
    end

    id_to_token = fill("", vocab_size + 1)
    id_to_token[PAD_ID + 1] = "<PAD>"
    id_to_token[EOS_ID + 1] = "<EOS>"
    id_to_token[UNK_ID + 1] = "<UNK>"
    for (c, tid) in char_map
        id_to_token[tid + 1] = string(c)
    end

    println("Saving to $output_file...")
    params = Dict(
        "PAD" => PAD_ID,
        "EOS" => EOS_ID,
        "UNK" => UNK_ID,
        "BLOCK_SIZE" => block_size,
        "VOCAB_SIZE" => vocab_size,
        "TOKENIZE_MODE" => "char",
    )
    jldsave(
        output_file;
        data=data_matrix,
        char_map=char_map,
        vocab=id_to_token,
        params=params,
    )
    save_metadata(meta_file; params=params, char_map=char_map, vocab=id_to_token)
    println("Done.")
end

function process_data_tokenizer(texts, output_file::AbstractString, meta_file::AbstractString, block_size::Int, tokenizer_name::AbstractString)
    tokenizer, vocab_size, PAD_ID, EOS_ID, UNK_ID, punct_ids, id_to_token = init_tokenizer(tokenizer_name)
    println("Vocabulary size: $vocab_size")

    data_matrix = Matrix{Int}(undef, block_size, 1024)
    col = 1
    for (idx, doc) in enumerate(texts)
        if idx % 1000 == 0
            println("Processing doc $idx / $(length(texts))")
        end

        s = normalize_text(doc)
        isempty(s) && continue

        ids = encode_ids(tokenizer, s)
        data_matrix, col = append_blocks_tokenizer!(data_matrix, col, ids, PAD_ID, EOS_ID, punct_ids, block_size)
    end

    data_matrix = data_matrix[:, 1:col-1]

    println("Saving to $output_file...")
    params = Dict(
        "PAD" => PAD_ID,
        "EOS" => EOS_ID,
        "UNK" => UNK_ID,
        "BLOCK_SIZE" => block_size,
        "VOCAB_SIZE" => vocab_size,
        "TOKENIZER_NAME" => tokenizer_name,
        "TOKENIZE_MODE" => "tokenizer",
    )
    jldsave(
        output_file;
        data=data_matrix,
        char_map=Dict{Any, Any}(),
        vocab=id_to_token,
        params=params,
    )
    save_metadata(meta_file; params=params, char_map=Dict{Any, Any}(), vocab=id_to_token)
    println("Done.")
end

function ensure_capacity!(data_matrix::Matrix{Int}, needed_cols::Int)
    if needed_cols <= size(data_matrix, 2)
        return data_matrix
    end
    new_cols = max(needed_cols, size(data_matrix, 2) * 2)
    out = Matrix{Int}(undef, size(data_matrix, 1), new_cols)
    out[:, 1:size(data_matrix, 2)] = data_matrix
    return out
end

function append_blocks_tokenizer!(data_matrix::Matrix{Int}, col::Int, ids::Vector{Int}, pad_id::Int, eos_id::Int, punct_ids::Set{Int}, block_size::Int)
    pos = 1
    while pos <= length(ids)
        window_end = min(pos + block_size - 1, length(ids))
        window = @view ids[pos:window_end]

        cut = 0
        for j in length(window):-1:1
            if window[j] in punct_ids
                cut = j
                break
            end
        end
        cut = cut == 0 ? length(window) : cut

        data_matrix = ensure_capacity!(data_matrix, col)
        @inbounds if cut >= block_size
            for j in 1:block_size
                data_matrix[j, col] = window[j]
            end
        else
            for j in 1:cut
                data_matrix[j, col] = window[j]
            end
            data_matrix[cut + 1, col] = eos_id
            for j in (cut + 2):block_size
                data_matrix[j, col] = pad_id
            end
        end
        col += 1
        pos += cut
    end
    return data_matrix, col
end

function process(cfg)
    if !isdir(cfg.data_dir)
        mkpath(cfg.data_dir)
    end

    files = readdir(cfg.data_dir; join=true)
    parquet_files = filter(f -> endswith(f, ".parquet"), files)
    file_path = if !isempty(cfg.parquet_file)
        cfg.parquet_file
    elseif !isempty(parquet_files)
        parquet_files[1]
    else
        error("No .parquet files found in $(cfg.data_dir). Please put the dataset under ./data or pass --parquet-file.")
    end

    texts = load_parquet_data(file_path)
    n_docs = length(texts)
    if cfg.max_docs > 0
        n_docs = min(n_docs, cfg.max_docs)
        texts = texts[1:n_docs]
    end
    println("Loaded $(length(texts)) documents.")

    mode = isempty(strip(cfg.tokenizer_name)) ? "char" : "tokenizer"
    println("Chunking data into blocks of size $(cfg.block_size)...")
    output_file = resolve_output_file(mode, cfg.data_dir, cfg.block_size, cfg.tokenizer_name; output_file=cfg.output_file)
    meta_file = resolve_meta_file(output_file; meta_file=cfg.meta_file)
    if mode == "char"
        process_data_char(texts, output_file, meta_file, cfg.block_size, cfg.char_vocab_docs)
    else
        process_data_tokenizer(texts, output_file, meta_file, cfg.block_size, cfg.tokenizer_name)
    end
    return
end

function data_prep(args::Vector{String})
    if "--help" in args || "-h" in args
        println("Usage: data [options]")
        println("Options:")
        println("  --data-dir <path>         Directory containing .parquet files (default: ./data)")
        println("  --parquet-file <path>     Specific parquet file to load")
        println("  --tokenizer-name <str>    HuggingFace tokenizer name (or empty for char-level)")
        println("  --output-file <path>      Output .jld2 file path")
        println("  --block-size <int>        Sequence length")
        println("  --max-docs <int>          Limit number of documents")
        return
    end
    cli = Utils.parse_cli_args(args)
    cfg = resolve_config(cli)
    process(cfg)
end

end # module
