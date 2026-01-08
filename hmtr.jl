#!/usr/bin/env julia
using Pkg
Pkg.activate(@__DIR__)

let opts = Base.JLOptions()
    println("hmtr.jl: JLOptions.compile_enabled=", hasfield(typeof(opts), :compile_enabled) ? getfield(opts, :compile_enabled) : missing)
    println("hmtr.jl: Julia cmd: ", Base.julia_cmd())
end

include("src/HMTR.jl")
using .HMTR

if abspath(PROGRAM_FILE) == @__FILE__
    HMTR.main(ARGS)
end
