using LinearAlgebra
using Random

include("misc.jl")
include("alg.jl")
include("plotting.jl")



### Here all methods may stuck in a local minimum, with objective not 0. Generate diff problems and check how many times which method converges for m = n = 100 and r = 50, it was easy to see.

name = "nmf"
Random.seed!(2023)

#   Gradient requires 3 matrix-matrix multiplications, objective - 1 and it can be reused for the next gradient.
function cost_lns(niter, nlns)
    N_total = (nlns - niter) / 3 + niter
    return N_total
end


RUN = true
RUN_lns = false
SAVE = false
LOAD = false
PLOTS = false

### Defining a problem



m, r, n = 100, 30, 100
name_instance = name * "_m=$(m)_n=$(n)_r=$r"

B = max.(randn(m, r), 0)
C = max.(randn(n, r), 0)
A = B * C'


f(X) = 0.5 * norm(X[1] * X[2]' - A)^2


function oracle_f(X)
    U, V = X[1], X[2]
    res = U * V' - A
    grad_U = res * V
    grad_V = res' * U
    return 0.5 * norm(res)^2, [grad_U, grad_V]
end

function oracle_f2!(X, grad)
    #U, V = X[1], X[2]
    res = X[1] * X[2]' - A
    mul!(grad[1], res, X[2])
    mul!(grad[2], res', X[1])
    return 0.5 * norm(res)^2, grad
end


function oracle_f3!(X, grad, res)
    #U, V = X[1], X[2]
    mul!(res, X[1], X[2]')
    res -= A
    mul!(grad[1], res, X[2])
    mul!(grad[2], res', X[1])
    #grad = res * V
    #grad_V = res' * U
    return 0.5 * norm(res)^2, grad
end


g(x) = 0.0
prox_g(x, α) = [max.(x[1], 0.0), max.(x[2], 0.0)]
#prox_g(x, α) = x

U0 = rand(m, r)
V0 = rand(n, r)
X0 = [U0, V0]

tol = 1e-5
N = 15000

if RUN

    @time X1, history1 = AdProxGrad(
        oracle_f,
        g,
        prox_g,
        X0;
        maxit = N,
        tol = tol,
        stop = "res",
        lns_init = true,
        verbose = true,
        ver = 2,
        track = ["res", "obj", "grad", "steps"],
        fixed_step = 1e-2,
    )


    @time X3, history3 = ProxGrad(
        oracle_f,
        g,
        prox_g,
        X0;
        maxit = N,
        tol = tol,
        stop = "res",
        lns_init = true,
        lns = true,
        verbose = true,
        track = ["res", "obj", "grad", "steps"],
        fixed_step = 1e-2,
        inc = 1.2,
        dcr = 0.1,
    )

end



incr = [1.1, 1.2, 1.5]
dcr = [0.5, 0.8, 0.9]
lns_param = Iterators.product(incr, dcr) |> collect
prox_grad_results = Dict()

if RUN_lns && ~LOAD
    @time X1, history1 = AdProxGrad(
        oracle_f,
        g,
        prox_g,
        X0;
        maxit = N,
        tol = tol,
        stop = "res",
        lns_init = true,
        verbose = true,
        ver = 2,
        track = ["res", "obj", "grad", "steps"],
        fixed_step = 0.001,
    )

    for (a, b) in [lns_param...]
        @time X3, history3, total = ProxGrad(
            oracle_f,
            g,
            prox_g,
            X0;
            maxit = N,
            tol = tol,
            stop = "res",
            lns_init = true,
            lns = true,
            verbose = true,
            track = ["res", "obj", "grad", "steps"],
            fixed_step = 1e-2,
            inc = a,
            dcr = b,
        )

        prox_grad_results[(a, b)] = (history3, total)
    end

end

if SAVE && ~LOAD
    save_data(history1, prox_grad_results, name_instance)
end

if LOAD
    history1, prox_grad_results = load_data(name_instance)
end

if PLOTS
    make_all_plots(history1, prox_grad_results, name_instance)
end
