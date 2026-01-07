#!/usr/bin/env julia
using Pkg
Pkg.activate(@__DIR__)

include("src/HMTR.jl")
using .HMTR

if abspath(PROGRAM_FILE) == @__FILE__
    HMTR.main(ARGS)
end
