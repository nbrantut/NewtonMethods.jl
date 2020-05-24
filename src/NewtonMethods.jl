module NewtonMethods

using ForwardDiff: jacobian
using LinearAlgebra: cond, I, inv

export newtonraphson, quasinewton, correlationmatrix

include("base.jl")

end
