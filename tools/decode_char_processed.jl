using JLD2

function parse_cli_args(args)
    out = Dict{Symbol, Any}()
    i = 1
    while i <= length(args)
        a = args[i]
        if startswith(a, "--")
            k, v = if occursin("=", a)
                parts = split(a[3:end], "=", limit=2)
                (parts[1], parts[2])
            else
                key = a[3:end]
                if i == length(args) || startswith(args[i + 1], "--")
                    (key, "true")
                else
                    i += 1
                    (key, args[i])
                end
            end
            out[Symbol(replace(k, "-" => "_"))] = v
        end
        i += 1
    end
    return out
end

function decode_char_processed(path::AbstractString; blocks::Int = 10, docs::Int = 0, start_col::Int = 1)
    d = JLD2.load(path)
    m = d["data"]
    params = d["params"]
    cm = d["char_map"]

    pad_id = params["PAD"]
    eos_id = params["EOS"]
    unk_id = params["UNK"]

    id_to_char = Dict{Int, Char}()
    for (c, tid64) in cm
        id_to_char[Int(tid64)] = c
    end

    fallback = Char(0x3f)

    decode_block = col -> begin
        buf = IOBuffer()
        for tid in col
            tid == pad_id && continue
            tid == eos_id && break
            if tid == unk_id
                print(buf, "?")
            else
                print(buf, get(id_to_char, tid, fallback))
            end
        end
        String(take!(buf))
    end

    if docs > 0
        printed = 0
        buf = IOBuffer()
        for j in start_col:size(m, 2)
            col = view(m, :, j)
            for tid in col
                tid == pad_id && continue
                if tid == eos_id
                    println(String(take!(buf)))
                    printed += 1
                    printed >= docs && return
                elseif tid == unk_id
                    print(buf, "?")
                else
                    print(buf, get(id_to_char, tid, fallback))
                end
            end
        end
        return
    end

    last_col = min(size(m, 2), start_col + blocks - 1)
    for j in start_col:last_col
        println(j, ": ", decode_block(view(m, :, j)))
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    cli = parse_cli_args(ARGS)
    input = get(cli, :input, get(cli, :path, ""))
    if isempty(String(input))
        println("Usage: julia tools/decode_char_processed.jl --input <processed.jld2> [--blocks N] [--docs N] [--start-col N]")
    else
        blocks = parse(Int, string(get(cli, :blocks, "10")))
        docs = parse(Int, string(get(cli, :docs, "0")))
        start_col = parse(Int, string(get(cli, :start_col, "1")))
        decode_char_processed(String(input); blocks=blocks, docs=docs, start_col=start_col)
    end
end

