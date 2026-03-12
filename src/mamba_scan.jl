function mamba_init_state(x, A, h_prev)
    d_model, L, batch = size(x)
    d_state = size(A, 2)
    if h_prev isa AbstractArray && ndims(h_prev) == 3 &&
       size(h_prev, 1) == d_model && size(h_prev, 2) == d_state && size(h_prev, 3) == batch
        return copy(h_prev)
    end
    h0 = similar(x, d_model, d_state, batch)
    fill!(h0, zero(eltype(h0)))
    return h0
end

function mamba_scan_sequential(x, dt_raw, B_raw, C_raw, A, D, h_prev=nothing)
    d_model, L, batch = size(x)
    d_state = size(A, 2)
    h = mamba_init_state(x, A, h_prev)

    D2 = reshape(D, d_model, 1)
    A2 = reshape(A, d_model, d_state, 1)

    if L == 0
        return similar(x, d_model, 0, batch), h
    end
    ys = similar(x, d_model, L, batch)

    dt_min = 1f-4
    dt_scale = 0.1f0
    for t in 1:L
        xt = view(x, :, t, :)
        dt_val = NNlib.softplus.(view(dt_raw, :, t, :)) .* dt_scale .+ dt_min
        dt = clamp.(dt_val, dt_min, 5f0)
        Bt = view(B_raw, :, t, :)
        Ct = view(C_raw, :, t, :)

        dt3 = reshape(dt, d_model, 1, batch)
        xt3 = reshape(xt, d_model, 1, batch)
        B3 = reshape(Bt, 1, d_state, batch)
        C3 = reshape(Ct, 1, d_state, batch)

        decay = exp.(A2 .* dt3)
        h .= h .* decay .+ (B3 .* dt3) .* xt3

        y = dropdims(sum(h .* C3; dims=2); dims=2) .+ (D2 .* xt)
        @views ys[:, t, :] .= y
    end

    return ys, h
end

function mamba_scan_sequential_with_state(x, dt_raw, B_raw, C_raw, A, D, h_prev=nothing)
    d_model, L, batch = size(x)
    d_state = size(A, 2)
    h = mamba_init_state(x, A, h_prev)

    D2 = reshape(D, d_model, 1)
    A2 = reshape(A, d_model, d_state, 1)

    ys = similar(x, d_model, L, batch)
    hs = similar(x, d_model, d_state, batch, L)

    dt_min = 1f-4
    dt_scale = 0.1f0
    for t in 1:L
        xt = @view x[:, t, :]
        dt_val = NNlib.softplus.(@view dt_raw[:, t, :]) .* dt_scale .+ dt_min
        dt = clamp.(dt_val, dt_min, 5f0)
        Bt = @view B_raw[:, t, :]
        Ct = @view C_raw[:, t, :]

        dt3 = reshape(dt, d_model, 1, batch)
        xt3 = reshape(xt, d_model, 1, batch)
        B3 = reshape(Bt, 1, d_state, batch)
        C3 = reshape(Ct, 1, d_state, batch)

        decay = exp.(A2 .* dt3)
        h .= h .* decay .+ (B3 .* dt3) .* xt3

        y = dropdims(sum(h .* C3; dims=2); dims=2) .+ (D2 .* xt)
        @views ys[:, t, :] .= y
        @views hs[:, :, :, t] .= h
    end

    return ys, h, hs
end

"""
    scan_log_forward_kernel!(S_out, log_P_out, log_a, u, stride, L)

CUDA Kernel for parallel associative scan log using single-thread sequential scan over time `L`.
This provides perfect scaling with zero allocation overhead compared to `cat`-based implementations.
"""
function scan_log_forward_kernel!(S_out, log_P_out, log_a, u, stride::Int, L::Int)
    # blockIdx().x is the unique flat index for d_model * d_state * batch
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    
    if idx <= stride
        acc_u = zero(eltype(S_out))
        acc_L = zero(eltype(log_P_out))
        
        # Traverse time dimension sequentially
        for t in 1:L
            # In column-major format with size (d_model, 1/d_state, batch, L),
            # the step to advance one time unit is precisely `stride`.
            lin_idx = idx + (t - 1) * stride 

            val_a = log_a[lin_idx]
            val_u = u[lin_idx]
            
            # (L2, u2) • (L1, u1)
            # new_u = u_t + exp(log_a_t) * u_{t-1}
            # new_L = log_a_t + L_{t-1}
            acc_u = val_u + exp(val_a) * acc_u
            acc_L = acc_L + val_a
            
            S_out[lin_idx] = acc_u
            log_P_out[lin_idx] = acc_L
        end
    end
    return nothing
end

function reverse_lastdim_kernel!(out, inp, stride::Int, L::Int, total::Int)
    lin = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if lin <= total
        idx = ((lin - 1) % stride) + 1
        t = ((lin - 1) ÷ stride) + 1
        src_lin = idx + (L - t) * stride
        out[lin] = inp[src_lin]
    end
    return nothing
end

function shift_right_zero_kernel!(out, inp, stride::Int, L::Int, total::Int)
    lin = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if lin <= total
        idx = ((lin - 1) % stride) + 1
        t = ((lin - 1) ÷ stride) + 1
        if t == 1
            out[lin] = zero(eltype(out))
        else
            src_lin = idx + (t - 2) * stride
            out[lin] = inp[src_lin]
        end
    end
    return nothing
end

function reverse_cumsum_kernel!(out, inp, stride::Int, L::Int)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= stride
        acc = zero(eltype(out))
        for t in L:-1:1
            lin = idx + (t - 1) * stride
            acc += inp[lin]
            out[lin] = acc
        end
    end
    return nothing
end

function reverse_lastdim_cuda(inp::CUDA.CuArray)
    out = similar(inp)
    stride = size(inp, 1) * size(inp, 2) * size(inp, 3)
    L = size(inp, 4)
    total = length(inp)
    threads = 256
    blocks = cld(total, threads)
    @cuda threads=threads blocks=blocks reverse_lastdim_kernel!(out, inp, stride, L, total)
    return out
end

function shift_right_zero_cuda(inp::CUDA.CuArray)
    out = similar(inp)
    stride = size(inp, 1) * size(inp, 2) * size(inp, 3)
    L = size(inp, 4)
    total = length(inp)
    threads = 256
    blocks = cld(total, threads)
    @cuda threads=threads blocks=blocks shift_right_zero_kernel!(out, inp, stride, L, total)
    return out
end

function reverse_cumsum_lastdim_cuda(inp::CUDA.CuArray)
    out = similar(inp)
    stride = size(inp, 1) * size(inp, 2) * size(inp, 3)
    L = size(inp, 4)
    threads = 256
    blocks = cld(stride, threads)
    @cuda threads=threads blocks=blocks reverse_cumsum_kernel!(out, inp, stride, L)
    return out
end

"""
    parallel_associative_scan_log(log_a::CUDA.CuArray, u::CUDA.CuArray)

High-performance GPU dispatch for parallel associative scan.
"""
function parallel_associative_scan_log(log_a::CUDA.CuArray, u::CUDA.CuArray)
    L_dim = ndims(u)
    L = size(u, L_dim)
    
    # Assert dimension 4 matching Mamba's usage
    @assert L_dim == 4 "Expected 4D arrays for log_a and u in GPU dispatch"

    stride = size(u, 1) * size(u, 2) * size(u, 3) # Number of independent sequences
    
    S_out = similar(u)
    log_P_out = similar(log_a)
    
    # Launch configuration
    threads = 256
    blocks = cld(stride, threads)
    
    # Launch Kernel
    @cuda threads=threads blocks=blocks scan_log_forward_kernel!(S_out, log_P_out, log_a, u, stride, L)
    
    return S_out, log_P_out
end

"""
    parallel_associative_scan_log(log_a, u)

CPU fallback and sequential associative scan via doubling algorithm (if L is reasonably small).
We can actually just use a simple sequential loop for CPU as it's faster than doubling with allocations.
"""
function parallel_associative_scan_log(log_a, u)
    L_dim = ndims(u)
    L = size(u, L_dim)
    
    # For CPU, sequential loop over the last dimension is often faster than cat allocations
    S_out = similar(u)
    log_P_out = similar(log_a)
    
    # Initial state
    pad_sz = collect(size(u))
    pad_sz[L_dim] = 1
    
    acc_u = zeros(eltype(u), Tuple(pad_sz))
    acc_L = zeros(eltype(log_a), Tuple(pad_sz))
    
    for t in 1:L
        val_a = selectdim(log_a, L_dim, t)
        val_u = selectdim(u, L_dim, t)
        
        acc_u .= val_u .+ exp.(val_a) .* acc_u
        acc_L .= acc_L .+ val_a
        
        selectdim(S_out, L_dim, t) .= acc_u
        selectdim(log_P_out, L_dim, t) .= acc_L
    end
    
    return S_out, log_P_out
end

function mamba_scan_parallel_with_state(x, dt_raw, B_raw, C_raw, A, D, h_prev=nothing)
    d_model, L, batch = size(x)
    d_state = size(A, 2)
    h0 = mamba_init_state(x, A, h_prev)

    if L == 0
        ys = similar(x, d_model, 0, batch)
        hs = similar(x, d_model, d_state, batch, 0)
        return ys, h0, hs
    end

    dt_min = 1f-4
    dt_scale = 0.1f0
    dt_val = NNlib.softplus.(dt_raw) .* dt_scale .+ dt_min
    dt = clamp.(dt_val, dt_min, 5f0)

    # Reshape for parallel processing
    # x: (d_model, L, batch) -> (d_model, 1, batch, L)
    x4 = reshape(permutedims(x, (1, 3, 2)), d_model, 1, batch, L)
    dt4 = reshape(permutedims(dt, (1, 3, 2)), d_model, 1, batch, L)
    B4 = reshape(permutedims(B_raw, (1, 3, 2)), 1, d_state, batch, L)
    C4 = reshape(permutedims(C_raw, (1, 3, 2)), 1, d_state, batch, L)

    A4 = reshape(A, d_model, d_state, 1, 1)
    
    # Compute log decays and u
    log_decay = A4 .* dt4
    u = x4 .* dt4 .* B4

    # --- STABLE PARALLEL SCAN ---
    # Replace the unstable "division path" with Associative Scan
    S_final, log_P_final = parallel_associative_scan_log(log_decay, u)
    
    # Compute final states
    # h_all = S_final + P * h0
    h0_4 = reshape(h0, d_model, d_state, batch, 1)
    P = exp.(log_P_final)
    h_all = S_final .+ P .* h0_4

    # Output projection
    y_tmp = dropdims(sum(h_all .* C4; dims=2); dims=2)
    y_seq = permutedims(y_tmp, (1, 3, 2))
    ys = y_seq .+ reshape(D, d_model, 1, 1) .* x

    h_last = copy(@view h_all[:, :, :, L])
    return ys, h_last, h_all
end

function mamba_scan_parallel(x, dt_raw, B_raw, C_raw, A, D, h_prev=nothing)
    ys, h_last, _ = mamba_scan_parallel_with_state(x, dt_raw, B_raw, C_raw, A, D, h_prev)
    return ys, h_last
end

function mamba_scan(x, dt_raw, B_raw, C_raw, A, D, h_prev=nothing, use_parallel::Bool=false)
    if use_parallel
        return mamba_scan_parallel(x, dt_raw, B_raw, C_raw, A, D, h_prev)
    else
        return mamba_scan_sequential(x, dt_raw, B_raw, C_raw, A, D, h_prev)
    end
end

function mamba_scan(x, dt_raw, B_raw, C_raw, A, D, h_prev=nothing; use_parallel::Bool=false)
    return mamba_scan(x, dt_raw, B_raw, C_raw, A, D, h_prev, use_parallel)
end

function mamba_scan_with_state(x, dt_raw, B_raw, C_raw, A, D, h_prev=nothing, use_parallel::Bool=false)
    if use_parallel
        return mamba_scan_parallel_with_state(x, dt_raw, B_raw, C_raw, A, D, h_prev)
    else
        return mamba_scan_sequential_with_state(x, dt_raw, B_raw, C_raw, A, D, h_prev)
    end
end

function mamba_scan_with_state(x, dt_raw, B_raw, C_raw, A, D, h_prev=nothing; use_parallel::Bool=false)
    return mamba_scan_with_state(x, dt_raw, B_raw, C_raw, A, D, h_prev, use_parallel)
end

function ChainRulesCore.rrule(::typeof(parallel_associative_scan_log), log_a, u)
    # Forward pass (no tracking, so mutation is fine)
    S_final, log_P_final = parallel_associative_scan_log(log_a, u)
    
    function parallel_associative_scan_log_pullback(Δ)
        if Δ isa ChainRulesCore.AbstractZero
            return (ChainRulesCore.NoTangent(), ChainRulesCore.ZeroTangent(), ChainRulesCore.ZeroTangent())
        end
        
        ΔS, ΔlogP = Δ
        # Handle cases where one of the gradients is Zero
        if ΔS isa ChainRulesCore.AbstractZero
            ΔS = similar(u)
            fill!(ΔS, zero(eltype(u)))
        end
        if ΔlogP isa ChainRulesCore.AbstractZero
            ΔlogP = similar(log_a)
            fill!(ΔlogP, zero(eltype(log_a)))
        end
        
        L = size(log_a, 4)
        use_cuda_kernels = CUDA.functional() && log_a isa CUDA.CuArray
        if use_cuda_kernels
            ΔS_rev = reverse_lastdim_cuda(ΔS)
            log_a_rev = reverse_lastdim_cuda(log_a)
            shifted_log_a_rev = shift_right_zero_cuda(log_a_rev)
            du_rev, _ = parallel_associative_scan_log(shifted_log_a_rev, ΔS_rev)
            du = reverse_lastdim_cuda(du_rev)
            S_prev = shift_right_zero_cuda(S_final)
            dlog_a = du .* S_prev .* exp.(log_a)
            dlog_a_from_P = reverse_cumsum_lastdim_cuda(ΔlogP)
        else
            ΔS_rev = reverse(ΔS, dims=4)
            log_a_rev = reverse(log_a, dims=4)
            shifted_log_a_rev = similar(log_a)
            @views selectdim(shifted_log_a_rev, 4, 1) .= zero(eltype(log_a))
            if L > 1
                @views selectdim(shifted_log_a_rev, 4, 2:L) .= selectdim(log_a_rev, 4, 1:L-1)
            end
            du_rev, _ = parallel_associative_scan_log(shifted_log_a_rev, ΔS_rev)
            du = reverse(du_rev, dims=4)
            S_prev = similar(S_final)
            @views selectdim(S_prev, 4, 1) .= zero(eltype(S_final))
            if L > 1
                @views selectdim(S_prev, 4, 2:L) .= selectdim(S_final, 4, 1:L-1)
            end
            dlog_a = du .* S_prev .* exp.(log_a)
            dlog_a_from_P = reverse(cumsum(reverse(ΔlogP, dims=4), dims=4), dims=4)
        end
        
        dlog_a = dlog_a .+ dlog_a_from_P
        
        return (ChainRulesCore.NoTangent(), dlog_a, du)
    end
    
    return (S_final, log_P_final), parallel_associative_scan_log_pullback
end
