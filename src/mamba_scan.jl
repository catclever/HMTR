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
    parallel_associative_scan_log(log_a, u)

Perform a parallel associative scan using the log-space operator:
(L2, u2) • (L1, u1) = (L2 + L1, exp(L2)*u1 + u2)

This avoids the numerical instability of the division path (u / P).
Operates along the last dimension.
"""
function parallel_associative_scan_log(log_a, u)
    L_dim = ndims(u)
    L = size(u, L_dim)
    
    # Force FP32 for accumulation stability if needed, 
    # but here we respect input types. Assuming inputs are Float32 as per user context.
    curr_log_a = copy(log_a)
    curr_u = copy(u)
    
    # Hillis-Steele Doubling Algorithm
    # Steps: ceil(log2(L))
    num_steps = ceil(Int, log2(L))
    
    for i in 0:(num_steps - 1)
        step = 2^i
        
        # Shifted views (right shift by step)
        # Pad with identity elements (0 for log_a, 0 for u)
        
        src_indices = 1:(L-step)
        
        # Slices
        src_log_a = selectdim(curr_log_a, L_dim, src_indices)
        src_u = selectdim(curr_u, L_dim, src_indices)
        
        # Padding
        pad_sz = collect(size(curr_u))
        pad_sz[L_dim] = step
        pad = similar(curr_u, Tuple(pad_sz))
        fill!(pad, zero(eltype(curr_u)))
        
        shifted_log_a = cat(pad, src_log_a; dims=L_dim)
        shifted_u = cat(pad, src_u; dims=L_dim)
        
        # Update
        # Operator: (curr_L, curr_u) • (shift_L, shift_u)
        # new_u = curr_u + exp(curr_L) * shift_u
        # new_L = curr_L + shift_L
        
        curr_u = curr_u .+ exp.(curr_log_a) .* shifted_u
        curr_log_a = curr_log_a .+ shifted_log_a
    end
    
    return curr_u, curr_log_a
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

function mamba_scan(x, dt_raw, B_raw, C_raw, A, D, h_prev=nothing; use_parallel::Bool=false)
    if use_parallel
        return mamba_scan_parallel(x, dt_raw, B_raw, C_raw, A, D, h_prev)
    else
        return mamba_scan_sequential(x, dt_raw, B_raw, C_raw, A, D, h_prev)
    end
end

function mamba_scan_with_state(x, dt_raw, B_raw, C_raw, A, D, h_prev=nothing; use_parallel::Bool=false)
    if use_parallel
        return mamba_scan_parallel_with_state(x, dt_raw, B_raw, C_raw, A, D, h_prev)
    else
        return mamba_scan_sequential_with_state(x, dt_raw, B_raw, C_raw, A, D, h_prev)
    end
end
