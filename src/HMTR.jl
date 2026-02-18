module HMTR

using Reexport
# using JET

# Export main entry points
export main
export train_entry, infer_entry, data_prep_entry, lint_entry, typecheck_entry

const PROJECT_ROOT = normpath(joinpath(@__DIR__, ".."))

# Include components
include("utils.jl")
@reexport using .Utils

include("model.jl")
@reexport using .Model

include("data.jl")
@reexport using .Data
include("train_stage1.jl")
@reexport using .TrainStage1
include("infer_stage1.jl")
@reexport using .InferStage1

include("ve/model.jl")
@reexport using .VEModel
include("ve/train.jl")
@reexport using .VETrainStage1
include("ve/infer.jl")
@reexport using .VEInferStage1
include("ve/data.jl")
@reexport using .VEData

function collect_julia_files(path::AbstractString)
    files = String[]
    if isfile(path)
        push!(files, path)
        return files
    end
    if isdir(path)
        for (root, _, fs) in walkdir(path)
            for f in fs
                endswith(f, ".jl") || continue
                push!(files, joinpath(root, f))
            end
        end
    end
    return sort(files)
end

function lint_entry(args::Vector{String})
    cli = Utils.parse_cli_args(args)
    root = string(get(cli, :path, joinpath(PROJECT_ROOT, "src")))
    files = collect_julia_files(root)
    n_errors = 0
    for f in files
        try
            Meta.parseall(read(f, String))
        catch e
            n_errors += 1
            println("LintError: $(f)")
            println(sprint(showerror, e))
        end
    end
    if n_errors == 0
        println("Lint OK: $(length(files)) files")
    else
        exit(1)
    end
end

function typecheck_entry(args::Vector{String})
    println("Typecheck skipped due to JET precompilation issues.")
end

function main(args::Vector{String})
    # Load .env at startup
    Utils.load_dotenv()

    if isempty(args)
        println("Usage: julia hmtr.jl [command] [options...]")
        println("\nCommands:")
        println("  train_stage1 (alias: train) - Train the Stage 1 AutoEncoder")
        println("  infer_stage1 (alias: infer) - Run inference/sampling with Stage 1 model")
        println("  train_stage1_ve (alias: train_ve) - Train the Stage 1 VE AutoEncoder")
        println("  infer_stage1_ve (alias: infer_ve) - Run inference/sampling with VE model")
        println("  data                        - Prepare data (Parquet -> JLD2)")
        println("  lint                        - Lint Julia files (syntax)")
        println("  typecheck                   - Typecheck Julia files (JET)")
        return
    end

    command = args[1]
    sub_args = args[2:end]

    if command == "train" || command == "train_stage1"
        if "--help" in sub_args || "-h" in sub_args
            TrainStage1.train_stage1(sub_args)
            return
        end
        TrainStage1.train_stage1(sub_args)
    elseif command == "infer" || command == "infer_stage1"
        InferStage1.infer_stage1(sub_args)
    elseif command == "train_ve" || command == "train_stage1_ve"
        if "--help" in sub_args || "-h" in sub_args
            VETrainStage1.train_stage1(sub_args)
            return
        end
        VETrainStage1.train_stage1(sub_args)
    elseif command == "infer_ve" || command == "infer_stage1_ve"
        VEInferStage1.infer_stage1(sub_args)
    elseif command == "data"
        Data.data_prep(sub_args)
    elseif command == "lint"
        lint_entry(sub_args)
    elseif command == "typecheck"
        typecheck_entry(sub_args)
    else
        println("Unknown command: $command")
        println("Available: train_stage1, infer_stage1, train_stage1_ve, infer_stage1_ve, data, lint, typecheck")
        exit(1)
    end
end

end # module
