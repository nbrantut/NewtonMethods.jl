module NewtonRaphson

using Zygote: gradient
using LinearAlgebra: cond

export jacobian, newtonraphson

include("base.jl")

end
