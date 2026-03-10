using Pkg
Pkg.activate(".")
include("src/HMTR.jl")
using .HMTR
using Dates

# Helper to find latest checkpoint
function find_latest_ckpt(dir)
    files = readdir(dir; join=true)
    jld2_files = filter(f -> endswith(f, ".jld2"), files)
    if isempty(jld2_files)
        return nothing
    end
    
    # Sort by mtime
    sort!(jld2_files, by=mtime, rev=true)
    return jld2_files[1]
end

ckpt_dir = "/home/HMTR/checkpoints"
latest_ckpt = find_latest_ckpt(ckpt_dir)

if latest_ckpt === nothing
    println("No checkpoints found in $ckpt_dir")
    exit(1)
end

println("Using latest checkpoint: $latest_ckpt")

# Test cases
test_cases = [
    "这是一个短句测试。",
    "这是一个非常长的中文句子，用于测试模型在处理长序列时的重构能力，特别是看它是否能够保持语义的连贯性和字符的完整性。",
    "在 2024 年，AI 技术取得了巨大的进步。We see improvements in LLMs.",
    "你好，世界！这是一个测试（全角括号）。",
    "白日依山尽，黄河入海流。欲穷千里目，更上一层楼。",
    "Short sentence test.",
    "Mixed: This is English. 这是中文。 12345. １２３４５.",
    "白日依山尽，黄河入海" # Incomplete sentence test
]

println("\n=== Starting Inference Tests ===")
println("Data File: /home/HMTR/data/processed_preserved_width.jld2")
println("Meta File: /home/HMTR/data/processed_preserved_width_meta.jld2")
println("Preserve Width: true")

for (i, text) in enumerate(test_cases)
    println("\n--- Test Case $i ---")
    args = [
        "--checkpoint-file", latest_ckpt,
        "--data-file", "/home/HMTR/data/processed_preserved_width.jld2",
        "--meta-file", "/home/HMTR/data/processed_preserved_width_meta.jld2",
        "--text", text,
        "--preserve-width", "1",
        "--force-cpu", "0" # Use GPU if available
    ]
    
    try
        HMTR.InferStage1.infer_stage1(args)
    catch e
        println("Error during inference: $e")
        # Base.showerror(stdout, e, catch_backtrace())
    end
end
println("\n=== Tests Completed ===")
