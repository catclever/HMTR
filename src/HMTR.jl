module HMTR

using Reexport

# Export main entry points
export main
export train_entry, infer_entry, data_prep_entry

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

function compiler_enabled()
    opts = Base.JLOptions()
    if hasfield(typeof(opts), :compile_enabled)
        return getfield(opts, :compile_enabled) != 0
    end
    return true
end

function compiler_supports_llvmcall()
    opts = Base.JLOptions()
    if hasfield(typeof(opts), :compile_enabled)
        mode = getfield(opts, :compile_enabled)
        return mode == 1 || mode == 2
    end
    return true
end

function relaunch_with_compiler(args::Vector{String})
    if get(ENV, "HMTR_RELAUNCHED_WITH_COMPILER", "0") == "1"
        return false
    end
    julia_path = joinpath(Sys.BINDIR, Base.julia_exename())
    project_arg = Base.active_project()
    project_flag = isempty(project_arg) ? "--project=$(PROJECT_ROOT)" : "--project=$(project_arg)"
    entry_script = joinpath(PROJECT_ROOT, "hmtr.jl")
    script_flag = isfile(entry_script) ? entry_script : ""

    if isempty(script_flag)
        error("Cannot find hmtr.jl entry script at $(entry_script).")
    end

    cmd = Cmd([julia_path, "--compile=all", project_flag, script_flag, args...])
    println("Re-launch cmd: ", cmd)
    withenv("HMTR_RELAUNCHED_WITH_COMPILER" => "1") do
        run(cmd)
    end
    return true
end

function main(args::Vector{String})
    # Load .env at startup
    Utils.load_dotenv()

    if isempty(args)
        println("Usage: julia hmtr.jl [command] [options...]")
        println("\nCommands:")
        println("  train_stage1 (alias: train) - Train the Stage 1 AutoEncoder")
        println("  infer_stage1 (alias: infer) - Run inference/sampling with Stage 1 model")
        println("  data                        - Prepare data (Parquet -> JLD2)")
        return
    end

    command = args[1]
    sub_args = args[2:end]

    if command == "train" || command == "train_stage1"
        if "--help" in sub_args || "-h" in sub_args
            TrainStage1.train_stage1(sub_args)
            return
        end
        opts = Base.JLOptions()
        println("JLOptions.compile_enabled=", hasfield(typeof(opts), :compile_enabled) ? getfield(opts, :compile_enabled) : missing)
        println("Julia cmd: ", Base.julia_cmd())
        if !compiler_supports_llvmcall()
            println("Training requires a compiler mode that supports `llvmcall` (not --compile=min/no).")
            println("Re-launching training in a compiler-enabled Julia process...")
            if relaunch_with_compiler(args)
                return
            end
        end
        try
            TrainStage1.train_stage1(sub_args)
        catch err
            if err isa ErrorException && occursin("`llvmcall` requires the compiler", err.msg)
                println("Training requires the Julia compiler, but this session cannot compile (`llvmcall` failed).")
                println("Re-launching training in a compiler-enabled Julia process...")
                if relaunch_with_compiler(args)
                    return
                end
            end
            rethrow()
        end
    elseif command == "infer" || command == "infer_stage1"
        InferStage1.infer_stage1(sub_args)
    elseif command == "data"
        Data.data_prep(sub_args)
    else
        println("Unknown command: $command")
        println("Available: train_stage1, infer_stage1, data")
        exit(1)
    end
end

end # module
