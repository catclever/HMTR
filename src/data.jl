module Data

using Parquet2
using DataFrames
using JLD2
using Random
if get(ENV, "JULIA_PYTHONCALL_EXE", "") == ""
    ENV["JULIA_PYTHONCALL_EXE"] = "python3"
end
using PythonCall
using Dates
using ..Utils

export data_prep

const PROJECT_ROOT = normpath(joinpath(@__DIR__, ".."))
const DEFAULT_DATA_DIR = get(ENV, "DATA_DIR", joinpath(PROJECT_ROOT, "data"))
const DEFAULT_PARQUET_FILE = get(ENV, "PARQUET_FILE", "")
const DEFAULT_TOKENIZER_NAME = get(ENV, "TOKENIZER_NAME", "")
const DEFAULT_MAX_DOCS = parse(Int, get(ENV, "MAX_DOCS", "0"))
const DEFAULT_CHAR_VOCAB_DOCS = parse(Int, get(ENV, "CHAR_VOCAB_DOCS", "10000"))
const DEFAULT_META_FILE = get(ENV, "META_FILE", "")

function resolve_config(cli::Dict{Symbol,Any})
    data_dir = string(get(cli, :data_dir, DEFAULT_DATA_DIR))
    parquet_file = string(get(cli, :parquet_file, DEFAULT_PARQUET_FILE))
    tokenizer_name = string(get(cli, :tokenizer_name, DEFAULT_TOKENIZER_NAME))
    max_docs = parse(Int, string(get(cli, :max_docs, DEFAULT_MAX_DOCS)))
    char_vocab_docs = parse(Int, string(get(cli, :char_vocab_docs, DEFAULT_CHAR_VOCAB_DOCS)))
    output_file = string(get(cli, :output_file, ""))
    meta_file = string(get(cli, :meta_file, DEFAULT_META_FILE))

    return (; data_dir, parquet_file, tokenizer_name, max_docs, char_vocab_docs, output_file, meta_file)
end

function resolve_output_file(mode::AbstractString, data_dir::AbstractString, tokenizer_name::AbstractString; output_file::AbstractString="")
    if !isempty(output_file)
        return output_file
    end
    ts = Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
    if mode != "char" && isempty(strip(tokenizer_name))
        error("--tokenizer-name is required for tokenizer mode")
    end
    label = mode == "char" ? "char" : "tok_" * sanitize_component(tokenizer_name)
    return joinpath(data_dir, "processed_stream_$(label)_$(ts).jld2")
end

function resolve_meta_file(output_file::AbstractString; meta_file::AbstractString="")
    if !isempty(meta_file)
        return meta_file
    end
    if endswith(output_file, ".jld2")
        base = output_file[1:end-5]
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
    # Define MASK ID - usually tokenizer.mask_token_id, or we pick one if not present
    mask_id = something(try_pyint(tokenizer.mask_token_id), unk_id)
    
    vocab_size = something(try_pyint(tokenizer.vocab_size), try_pyint(length(tokenizer)), 0)

    vocab_py = tokenizer.get_vocab()
    vocab_jl = pyconvert(Dict{String,Int}, vocab_py)
    max_id = maximum(values(vocab_jl))
    id_to_token = fill("", max(vocab_size, max_id + 1))
    for (tok, tid) in vocab_jl
        id_to_token[tid+1] = tok
    end

    return tokenizer, vocab_size, pad_id, eos_id, unk_id, mask_id, id_to_token
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

function encode_chars(s::String, char_map, unk_id::Int)
    out = Vector{Int}(undef, length(s))
    i = 1
    for c in s
        out[i] = get(char_map, c, unk_id)
        i += 1
    end
    return out
end

function process_data_char_stream(texts, output_file::AbstractString, meta_file::AbstractString, char_vocab_docs::Int)
    sample_texts = texts[1:min(end, char_vocab_docs)]
    vocab_chars = sort!(collect(build_char_set(sample_texts)))

    # 1=PAD, 2=EOS, 3=UNK, 4=MASK
    char_map = Dict{Char,Int}(c => i + 4 for (i, c) in enumerate(vocab_chars))
    PAD_ID = 1
    EOS_ID = 2
    UNK_ID = 3
    MASK_ID = 4
    vocab_size = length(char_map) + 4

    println("Vocabulary size: $vocab_size")

    # Arrays to store the continuous stream
    # chunks needed because we don't know total size yet
    all_tokens = Int[]
    reset_flags = Bool[]

    total_docs = length(texts)
    for (idx, doc) in enumerate(texts)
        if idx % 1000 == 0
            println("Processing doc $idx / $total_docs")
        end

        s = normalize_text(doc)
        isempty(s) && continue

        # No sentence splitting! Continuous document.
        ids = encode_chars(s, char_map, UNK_ID)
        if isempty(ids)
            continue
        end

        # Append EOS at end of document
        push!(ids, EOS_ID)

        # Mark reset at the BEGINNING of this document segment
        # First token of this doc gets reset=true
        doc_resets = falses(length(ids))
        doc_resets[1] = true

        append!(all_tokens, ids)
        append!(reset_flags, doc_resets)
    end

    id_to_token = fill("", vocab_size + 1)
    id_to_token[PAD_ID+1] = "<PAD>"
    id_to_token[EOS_ID+1] = "<EOS>"
    id_to_token[UNK_ID+1] = "<UNK>"
    id_to_token[MASK_ID+1] = "<MASK>"
    for (c, tid) in char_map
        id_to_token[tid+1] = string(c)
    end

    println("Total tokens: $(length(all_tokens))")
    println("Saving to $output_file...")
    params = Dict(
        "PAD" => PAD_ID,
        "EOS" => EOS_ID,
        "UNK" => UNK_ID,
        "MASK" => MASK_ID,
        "VOCAB_SIZE" => vocab_size,
        "TOKENIZE_MODE" => "char",
        "STREAM_MODE" => true
    )

    jldsave(
        output_file;
        tokens=all_tokens,
        reset_flags=reset_flags,
        char_map=char_map,
        vocab=id_to_token,
        params=params,
    )
    save_metadata(meta_file; params=params, char_map=char_map, vocab=id_to_token)
    println("Done.")
end

function process_data_tokenizer_stream(texts, output_file::AbstractString, meta_file::AbstractString, tokenizer_name::AbstractString)
    tokenizer, vocab_size, PAD_ID, EOS_ID, UNK_ID, MASK_ID, id_to_token = init_tokenizer(tokenizer_name)
    
    # If MASK_ID was UNK (-1), force assign one if space allows? 
    # Or rely on tokenizer having one.
    if MASK_ID == -1
        # Fallback: create a custom ID at the end of vocab?
        MASK_ID = vocab_size # Extend vocab by 1
        vocab_size += 1
        push!(id_to_token, "<MASK>")
    end

    println("Vocabulary size: $vocab_size")

    all_tokens = Int[]
    reset_flags = Bool[]

    total_docs = length(texts)
    for (idx, doc) in enumerate(texts)
        if idx % 1000 == 0
            println("Processing doc $idx / $total_docs")
        end

        s = normalize_text(doc)
        isempty(s) && continue

        ids = encode_ids(tokenizer, s)
        if isempty(ids)
            continue
        end

        # Append EOS
        if ids[end] != EOS_ID
             push!(ids, EOS_ID)
        end

        doc_resets = falses(length(ids))
        doc_resets[1] = true

        append!(all_tokens, ids)
        append!(reset_flags, doc_resets)
    end

    println("Total tokens: $(length(all_tokens))")
    println("Saving to $output_file...")
    params = Dict(
        "PAD" => PAD_ID,
        "EOS" => EOS_ID,
        "UNK" => UNK_ID,
        "MASK" => MASK_ID,
        "VOCAB_SIZE" => vocab_size,
        "TOKENIZER_NAME" => tokenizer_name,
        "TOKENIZE_MODE" => "tokenizer",
        "STREAM_MODE" => true
    )

    jldsave(
        output_file;
        tokens=all_tokens,
        reset_flags=reset_flags,
        char_map=Dict{Any,Any}(),
        vocab=id_to_token,
        params=params,
    )
    save_metadata(meta_file; params=params, char_map=Dict{Any,Any}(), vocab=id_to_token)
    println("Done.")
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
    println("Processing data in Continuous Stream mode...")
    
    output_file = resolve_output_file(mode, cfg.data_dir, cfg.tokenizer_name; output_file=cfg.output_file)
    meta_file = resolve_meta_file(output_file; meta_file=cfg.meta_file)
    
    if mode == "char"
        process_data_char_stream(texts, output_file, meta_file, cfg.char_vocab_docs)
    else
        process_data_tokenizer_stream(texts, output_file, meta_file, cfg.tokenizer_name)
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
        println("  --max-docs <int>          Limit number of documents")
        return
    end
    cli = Utils.parse_cli_args(args)
    cfg = resolve_config(cli)
    process(cfg)
end

end # module
