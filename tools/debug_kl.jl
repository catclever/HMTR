using Lux, JLD2, CUDA, Statistics, Random
include("../src/HMTR.jl")
using .HMTR
using .HMTR.Model: HMTR_Stage1_AutoEncoder
using .HMTR.Data: encode_text_to_seq

# 1. Load Model
ckpt_dir = "/home/HMTR/checkpoints"
files = readdir(ckpt_dir; join=true)
stage1_files = filter(f -> occursin("ckpt_pad_aware_stage1", f) && endswith(f, ".jld2"), files)
sort!(stage1_files, by=mtime, rev=true)
latest_ckpt = stage1_files[1]
println("Loading: $latest_ckpt")

ckpt = JLD2.load(latest_ckpt)
ps = ckpt["ps"]
st = ckpt["st"]
meta = JLD2.load("/home/HMTR/data/processed_preserved_width_meta.jld2")

# Model setup
dim = size(ps.encoder.embedding.weight, 1)
vocab_size = meta["params"]["VOCAB_SIZE"]
mamba_d_state = 16 # Default
if haskey(ckpt, "mamba_d_state")
    mamba_d_state = Int(ckpt["mamba_d_state"])
end

model = HMTR_Stage1_AutoEncoder(vocab_size, dim; block_size=8, pad_id=meta["params"]["PAD"], eos_id=meta["params"]["EOS"], mamba_d_state=mamba_d_state)
dev = cpu_device() # Debug on CPU for inspection
ps = ps |> dev
st = st |> dev

# 2. Forward Pass on a sample
text = "这是一个测试句子，用于分析KL散度的构成。"
x = HMTR.Data.encode_text_to_seq(text, (params=meta["params"], char_map=meta["char_map"]); preserve_width=true)
x_dev = x |> dev

# Encoder Forward
capsules, _st_enc = model.encoder(x_dev, ps.encoder, st.encoder)
mu, logvar = capsules

# 3. Analyze Statistics
mu_mean = mean(mu)
mu_sq_mean = mean(abs2, mu)
logvar_mean = mean(logvar)
var_mean = mean(exp.(logvar))
sigma_mean = mean(exp.(0.5f0 .* logvar))

println("\n=== Latent Statistics ===")
println("mu_mean: ", mu_mean)
println("mu_sq_mean (E[μ²]): ", mu_sq_mean)
println("logvar_mean: ", logvar_mean)
println("var_mean (E[σ²]): ", var_mean)
println("sigma_mean (E[σ]): ", sigma_mean)

# 4. Calculate KL Breakdown
# KL = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
#    = -0.5 * [ 1 + logvar_mean - mu_sq_mean - var_mean ] * Dim
# Let's look at per-dimension term first (inside sum)

term_1 = 1.0
term_logvar = logvar_mean
term_mu_sq = -mu_sq_mean
term_var = -var_mean

sum_inner = term_1 + term_logvar + term_mu_sq + term_var
kl_per_dim = -0.5 * sum_inner
kl_total = kl_per_dim * dim # Sum over dimensions

println("\n=== KL Breakdown (Approx per dim) ===")
println("Term 1 (Constant): ", term_1)
println("Term 2 (logvar):   ", term_logvar)
println("Term 3 (-mu^2):    ", term_mu_sq)
println("Term 4 (-var):     ", term_var)
println("-------------------------")
println("Sum Inner:         ", sum_inner)
println("KL per dim:        ", kl_per_dim)
println("Total KL (x$dim):    ", kl_total)

# 5. Check exact KL calculation from code
kl_per_element = -0.5f0 .* (1f0 .+ logvar .- abs2.(mu) .- exp.(logvar))
kl_total_exact = sum(kl_per_element) # Sum over all dims and batch
println("\n=== Exact KL from formula ===")
println("Total KL (sum): ", kl_total_exact)
println("Mean KL (per batch): ", kl_total_exact / size(x, 2))
