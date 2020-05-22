"""
    jacobian(f, x)

Return jacobian matrix of operator f at point x: Jᵢⱼ = ∂fᵢ/∂xⱼ

Uses Zygote for automatic differentiation.

Copied directly from https://github.com/FluxML/Zygote.jl/issues/98

Written by github user gdkrmr.
"""
function jacobian(f, x)
    y = f(x)
    n = length(y)
    m = length(x)
    T = eltype(y)
    j = Array{T, 2}(undef, n, m)
    for i in 1:n
        j[i, :] .= gradient(x -> f(x)[i], x)[1]
    end
    return j
end

"""
	funcjac(f, x)

Return value of f and jacobian together.
"""
function funcjac(f, x)
    y = f(x)
    n = length(y)
    m = length(x)
    T = eltype(y)
    j = Array{T, 2}(undef, n, m)
    for i in 1:n
        j[i, :] .= gradient(x -> f(x)[i], x)[1]
    end
    return y, j
end

"""
    newtonraphson(f::Function, x0::Number, fprime::Function, args::Tuple(); tol=1e-8, maxiter=50, eps0=1e-10)

Find solution of f(x) = 0 (x is scalar) using Newton-Raphson iterations. 

Copied from https://mmas.github.io/newton-julia

"""
function newtonraphson(f::Function, x0::Number, fprime::Function, args::Tuple=();
                tol::AbstractFloat=1e-8, maxiter::Integer=50, eps0::AbstractFloat=1e-10)
    for _ in 1:maxiter
        yprime = fprime(x0, args...)
        if abs(yprime) < eps0
            warn("First derivative is zero")
            return x0
        end
        y = f(x0, args...)
        x1 = x0 - y/yprime
        if abs(x1-x0) < tol
            return x1
        end
        x0 = x1
    end
    error("Max iteration exceeded")
end


"""
    newtonraphson(f,x0,jac,args; tol=1e-8, maxiter=50, eps0=1e-10)

NR solution for multidimensional problem f(x)=0. Requires jacobian.

"""
function newtonraphson(f::Function, x0::AbstractVector, jac::Function, args::Tuple=(); tol::AbstractFloat=1e-8, maxiter::Integer=50, eps0::AbstractFloat=1e-10)
    for _ in 1:maxiter
        J = jac(x0, args...)
        if cond(J) > 1/eps0
            warn("Jacobian is ill-conditioned")
            return x0
        end
        y = f(x0, args...)
        x1 = x0 - J\y
        if maximum(abs.(x1-x0)) < tol
            return x1
        end
        x0 = x1
    end
    error("Max iteration exceeded")
end

"""
    newtonraphson(f,x0,args; tol=1e-8, maxiter=50, eps0=1e-10)

    Method without explicit jacobian, using automatic differentiation.
"""
function newtonraphson(f::Function, x0::AbstractVector, args::Tuple=(); tol::AbstractFloat=1e-8, maxiter::Integer=50, eps0::AbstractFloat=1e-10)
    for _ in 1:maxiter
        y, J = funcjac(x->f(x,args...), x0)
        if cond(J) > 1/eps0
            warn("Jacobian is ill-conditioned")
            return x0
        end
        x1 = x0 - J\y
        if maximum(abs.(x1-x0)) < tol
            return x1
        end
        x0 = x1
    end
    error("Max iteration exceeded")
end
