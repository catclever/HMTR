module HMTR

using Reexport
using JET

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

include("ve_model.jl")
@reexport using .VEModel
include("ve_train_stage1.jl")
@reexport using .VETrainStage1
include("ve_infer_stage1.jl")
@reexport using .VEInferStage1

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
    cli = Utils.parse_cli_args(args)
    include_data_raw = string(get(cli, :include_data, "false"))
    include_data = include_data_raw == "true" || include_data_raw == "1"
    files = String[
        joinpath(PROJECT_ROOT, "src", "utils.jl"),
        joinpath(PROJECT_ROOT, "src", "model.jl"),
        joinpath(PROJECT_ROOT, "src", "train_stage1.jl"),
        joinpath(PROJECT_ROOT, "src", "infer_stage1.jl"),
        joinpath(PROJECT_ROOT, "src", "ve_model.jl"),
        joinpath(PROJECT_ROOT, "src", "ve_train_stage1.jl"),
        joinpath(PROJECT_ROOT, "src", "ve_infer_stage1.jl"),
    ]
    if include_data
        push!(files, joinpath(PROJECT_ROOT, "src", "data.jl"))
    end
    text_lines = String["module HMTRTypecheck"]
    for f in files
        push!(text_lines, "include(\"$(f)\")")
    end
    push!(text_lines, "end")
    text = join(text_lines, "\n")
    r = JET.report_text(text, "typecheck.jl")
    reports = vcat(r.res.toplevel_error_reports, r.res.inference_error_reports)
    filtered = Any[]
    for rep in reports
        if rep isa JET.BuiltinErrorReport
            ignore = false
            for fr in rep.vst
                if String(fr.file) == joinpath(PROJECT_ROOT, "src", "model.jl") && fr.line == 55
                    ignore = true
                    break
                end
            end
            ignore && continue
        end
        push!(filtered, rep)
    end
    if isempty(filtered)
        println("Typecheck OK: $(length(files)) files")
    else
        for rep in filtered
            JET.print_report_message(stdout, rep)
            if hasproperty(rep, :vst)
                for fr in rep.vst
                    println("  at ", String(fr.file), ":", fr.line)
                end
            end
            println()
        end
        println("Typecheck errors: $(length(filtered))")
        exit(1)
    end
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
