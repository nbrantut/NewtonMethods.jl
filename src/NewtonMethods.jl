module NewtonMethods

using ForwardDiff: jacobian
using LinearAlgebra: cond

export newtonraphson, quasinewton

include("base.jl")

end
