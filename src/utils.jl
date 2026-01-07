module Utils

using Dates
using Random

export load_dotenv, parse_cli_args, get_env_int, get_env_float, get_env_str, sanitize_component

function load_dotenv(path::AbstractString = ".env"; override::Bool = false)
    if !isfile(path)
        return
    end
    for raw_line in eachline(path)
        line = strip(raw_line)
        isempty(line) && continue
        startswith(line, "#") && continue

        if startswith(line, "export ")
            line = strip(line[8:end])
        end

        eq = findfirst(==('='), line)
        eq === nothing && continue

        key = strip(line[1:eq - 1])
        isempty(key) && continue

        value = strip(line[eq + 1:end])
        if (startswith(value, "\"") && endswith(value, "\"")) || (startswith(value, "'") && endswith(value, "'"))
            value = value[2:end - 1]
        end

        if !override && haskey(ENV, key)
            continue
        end
        ENV[key] = value
    end
end

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

function get_env_int(key::String, default::String)
    parse(Int, get(ENV, key, default))
end

function get_env_float(key::String, default::String)
    parse(Float64, get(ENV, key, default))
end

function get_env_str(key::String, default::String)
    get(ENV, key, default)
end

function sanitize_component(s::AbstractString)
    return replace(lowercase(s), r"[^a-z0-9._-]+" => "-")
end

end # module
