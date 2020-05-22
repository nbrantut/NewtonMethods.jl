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
    xc = copy(x0)
    for _ in 1:maxiter
        J = jac(xc, args...)
        if cond(J) > 1/eps0
            warn("Jacobian is ill-conditioned")
            return xc
        end
        y = f(xc, args...)
        x1 = xc - J\y
        if maximum(abs.(x1-xc)) < tol
            return x1
        end
        xc = x1
    end
    error("Max iteration exceeded")
end

"""
    newtonraphson(f,x0,args; tol=1e-8, maxiter=50, eps0=1e-10)

    Method without explicit jacobian, using automatic differentiation.
"""
function newtonraphson(f::Function, x0::AbstractVector, args::Tuple=(); tol::AbstractFloat=1e-8, maxiter::Integer=50, eps0::AbstractFloat=1e-10)
    xc = copy(x0)
    for _ in 1:maxiter
        J = jacobian(x->f(x,args...), xc)
        if cond(J) > 1/eps0
            warn("Jacobian is ill-conditioned")
            return xc
        end
        y = f(xc, args...)
        x1 = xc - J\y
        if maximum(abs.(x1-xc)) < tol
            return x1
        end
        xc = x1
    end
    error("Max iteration exceeded")
end

function quasinewton(dobs::AbstractVector, g::Function,
                     mprior::AbstractVector, jac::Function, args::Tuple(),
                     CMi::AbstractMatrix, CDi::AbstractMatrix;
                     step::AbstractFloat=1.0, tol::AbstractFloat=1e-8,
                     maxiter::Integer=50, eps0::AbstractFloat=1e-10)

    m = copy(mprior)
    
    for _ in 1:maxiter
        G = jac(m, args...)
        A = G'*CDi*G + CMi
        if cond(A)>1/eps0
            warn("Operator is ill conditioned. Consider Smoothing.")
            return xc
        end
        d = g(m, args...)
        b = G'*CDi*(d - dobs) + CMi*(m-mprior)
        dm = -step*(A\b)
        m = m+dm
        if maximum(abs.(dm)) < tol
            return m
        end
    end
    error("Max iteration exceeded")
    
end
