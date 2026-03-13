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
    scan_log_single_pass_kernel!(u_out, log_out, log_a, u, stride, L)

CUDA Kernel for parallel associative scan log using single-thread sequential scan per sequence.
Each thread processes one independent sequence along the time dimension `L`.
This avoids O(log L) kernel launches and O(L log L) global memory traffic,
providing O(L) complexity and much better performance for large batch/d_model.
"""
function scan_log_single_pass_kernel!(u_out, log_out, log_a, u, stride::Int, L::Int)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    
    if idx <= stride
        # Initial state at t=1
        curr_u = u[idx]
        curr_log = log_a[idx]
        
        u_out[idx] = curr_u
        log_out[idx] = curr_log
        
        for t in 2:L
            lin_idx = idx + (t - 1) * stride
            
            val_u = u[lin_idx]
            val_log = log_a[lin_idx]
            
            # (L2, u2) • (L1, u1) = (L2 + L1, exp(L2)u1 + u2)
            # Accumulate: curr = val • curr
            curr_u = val_u + exp(val_log) * curr_u
            curr_log = val_log + curr_log
            
            u_out[lin_idx] = curr_u
            log_out[lin_idx] = curr_log
        end
    end
    return nothing
end

function scan_log_hillis_steele_step_kernel!(u_next, log_next, u_prev, log_prev, stride::Int, L::Int, offset::Int, total::Int)
    tid = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if tid <= total
        t = ((tid - 1) ÷ stride) + 1
        if t > offset
            seq_idx = ((tid - 1) % stride) + 1
            prev_t = t - offset
            prev_lin = seq_idx + (prev_t - 1) * stride
            curr_u = u_prev[tid]
            curr_log = log_prev[tid]
            prev_u = u_prev[prev_lin]
            prev_log = log_prev[prev_lin]
            u_next[tid] = curr_u + exp(curr_log) * prev_u
            log_next[tid] = curr_log + prev_log
        else
            u_next[tid] = u_prev[tid]
            log_next[tid] = log_prev[tid]
        end
    end
    return nothing
end

"""
    scan_log_single_pass_backward_kernel!(du_out, dlog_out, dS, dlogP, S_final, log_a, stride, L)

Fused backward kernel for associative scan.
Computes gradients for `u` and `log_a` in a single reverse pass over time.
Avoids multiple temporary allocations and kernel launches.
"""
function scan_log_single_pass_backward_kernel!(du_out, dlog_out, dS, dlogP, S_final, log_a, stride::Int, L::Int)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    
    if idx <= stride
        acc_delta = zero(eltype(dS))
        acc_gamma = zero(eltype(dlogP))
        
        # Traverse time backwards from L down to 1
        for t in L:-1:1
            lin_idx = idx + (t - 1) * stride
            
            # Load gradients
            val_dS = dS[lin_idx]
            val_dlogP = dlogP[lin_idx]
            
            # Update gamma (reverse suffix sum of dlogP)
            acc_gamma += val_dlogP
            
            # Update delta
            # delta_t = dS_t + exp(log_a_{t+1}) * delta_{t+1}
            if t < L
                next_lin_idx = lin_idx + stride
                log_a_next = log_a[next_lin_idx]
                acc_delta = val_dS + exp(log_a_next) * acc_delta
            else
                acc_delta = val_dS
            end
            
            # Compute du_t = delta_t
            du_out[lin_idx] = acc_delta
            
            # Compute dlog_a_t
            # dlog_a_t = delta_t * exp(log_a_t) * S_{t-1} + acc_gamma
            term1 = zero(eltype(dS))
            if t > 1
                prev_lin_idx = lin_idx - stride
                S_prev = S_final[prev_lin_idx]
                log_a_curr = log_a[lin_idx]
                term1 = acc_delta * exp(log_a_curr) * S_prev
            end
            
            dlog_out[lin_idx] = term1 + acc_gamma
        end
    end
    return nothing
end

"""
    parallel_associative_scan_log(log_a::CUDA.CuArray, u::CUDA.CuArray)

High-performance GPU dispatch for parallel associative scan using "One Thread Per Sequence" strategy.
This is significantly faster than Hillis-Steele for large batch sizes.
"""
function parallel_associative_scan_log(log_a::CUDA.CuArray, u::CUDA.CuArray)
    L_dim = ndims(u)
    L = size(u, L_dim)
    
    @assert L_dim == 4 "Expected 4D arrays for log_a and u in GPU dispatch"
    if L <= 1
        return copy(u), copy(log_a)
    end

    stride = size(u, 1) * size(u, 2) * size(u, 3)
    impl = lowercase(get(ENV, "MAMBA_SCAN_IMPL", "single_pass"))

    if impl == "time_parallel" || impl == "hillis_steele"
        total = stride * L
        threads = 256
        blocks = cld(total, threads)
        u_prev = copy(u)
        log_prev = copy(log_a)
        u_next = similar(u)
        log_next = similar(log_a)
        offset = 1
        while offset < L
            @cuda threads=threads blocks=blocks scan_log_hillis_steele_step_kernel!(
                u_next, log_next, u_prev, log_prev, stride, L, offset, total
            )
            u_prev, u_next = u_next, u_prev
            log_prev, log_next = log_next, log_prev
            offset <<= 1
        end
        return u_prev, log_prev
    else
        u_out = similar(u)
        log_out = similar(log_a)
        threads = 256
        blocks = cld(stride, threads)
        @cuda threads=threads blocks=blocks scan_log_single_pass_kernel!(u_out, log_out, log_a, u, stride, L)
        return u_out, log_out
    end
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
            du = similar(u)
            dlog_a = similar(log_a)
            stride = size(u, 1) * size(u, 2) * size(u, 3)
            threads = 256
            blocks = cld(stride, threads)
            
            @cuda threads=threads blocks=blocks scan_log_single_pass_backward_kernel!(
                du, dlog_a, ΔS, ΔlogP, S_final, log_a, stride, L
            )
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
            dlog_a = dlog_a .+ dlog_a_from_P
        end
        
        return (ChainRulesCore.NoTangent(), dlog_a, du)
    end
    
    return (S_final, log_P_final), parallel_associative_scan_log_pullback
end
