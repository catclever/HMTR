module PrecisionRuntime

using Lux
using LinearAlgebra

export install_bf16_gpu_dense_fallback!

const _BF16_GPU_DENSE_FALLBACK_INSTALLED = Ref(false)

function Lux.LuxLib.Impl.matmuladd!(
    C::AbstractMatrix{Core.BFloat16},
    ::Lux.LuxLib.GPUBroadcastOp{Lux.MLDataDevices.CUDADevice},
    A::AbstractMatrix{Core.BFloat16},
    B::AbstractMatrix{Core.BFloat16},
    bias::AbstractVector{Core.BFloat16},
)
    C .= bias
    LinearAlgebra.mul!(C, A, B, true, true)
    return nothing
end

function Lux.LuxLib.Impl.fused_dense!(
    y::AbstractMatrix{Core.BFloat16},
    opmode::Lux.LuxLib.GPUBroadcastOp{Lux.MLDataDevices.CUDADevice},
    act::F,
    weight::AbstractMatrix{Core.BFloat16},
    x::AbstractMatrix{Core.BFloat16},
    b::Union{Nothing,AbstractVector{Core.BFloat16}},
) where {F}
    Lux.LuxLib.Impl.matmul!(y, opmode, weight, x)
    Lux.LuxLib.Impl.bias_activation!(y, opmode, act, y, b)
    return nothing
end

function Lux.LuxLib.Impl.cublasLt_fused_dense!(
    z::AbstractMatrix{Core.BFloat16},
    act::F,
    weight::AbstractMatrix{Core.BFloat16},
    x::AbstractMatrix{Core.BFloat16},
    b::Union{Nothing,AbstractVector{Core.BFloat16}},
) where {F}
    LinearAlgebra.mul!(z, weight, x)
    if b === nothing
        broadcast!(act, z, z)
    else
        broadcast!(act ∘ +, z, z, reshape(b, :, 1))
    end
    return nothing
end

function Lux.LuxLib.Impl.cublasLt_fused_dense!(
    z::AbstractMatrix{Core.BFloat16},
    act::F,
    weight::AbstractMatrix{Core.BFloat16},
    x::AbstractMatrix{Core.BFloat16},
    b::Union{Nothing,AbstractVector{Core.BFloat16}},
    y::AbstractMatrix{Core.BFloat16},
) where {F}
    LinearAlgebra.mul!(y, weight, x)
    if b === nothing
        broadcast!(act, y, y)
    else
        broadcast!(act ∘ +, y, y, reshape(b, :, 1))
    end
    broadcast!(act, z, y)
    return nothing
end

function install_bf16_gpu_dense_fallback!()
    _BF16_GPU_DENSE_FALLBACK_INSTALLED[] && return nothing
    _BF16_GPU_DENSE_FALLBACK_INSTALLED[] = true
    return nothing
end

end
