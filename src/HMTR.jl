module HMTR

using Reexport

# Export main entry points
export main
export train_entry, infer_entry, data_prep_entry

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
        TrainStage1.train_stage1(sub_args)
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
