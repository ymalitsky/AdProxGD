using LinearAlgebra
using ForwardDiff: gradient
using Random

include("misc.jl")
include("alg.jl")
include("plotting.jl")


# Problem source: check in B & V, Chapter 5.1.6
name = "dual_max_ent"

# gradient requires 2 matrix-vector products, one of which is reused for the objective and vice versa. Prox are ignored
function cost_lns(niter, nlns)
    N_total = niter + Int(ceil((nlns - niter) / 2))
    return N_total
end


RUN_lns = false # to run prox_grad with linesearch with various parameters
SAVE = false
LOAD = false
PLOTS = false
Random.seed!(2023)

### Defining a problem

m, n = 500, 100
name_instance = name * "_m=$(m)_n=$(n)_normal"

A = randn(m, n)


x_feas = rand(n)
x_feas = x_feas ./ norm(x_feas, 1)
b = A * x_feas

function f_dual(y)
    la = y[1:end-1]
    nu = y[end]
    return dot(b, la) + nu + exp.(-nu - 1) * sum(exp.(-A' * la))
end

df_dual(y) = gradient(f_dual, y)

function oracle_f(y)
    return f_dual(y), df_dual(y)
end

g(y) = 0
function prox_g(y, alpha)
    y[1:end-1] = max.(0, y[1:end-1])
    return y
end

### Running algorithms


y0 = (zeros(m + 1))
tol = 1e-6
N = 10000
if RUN
    y1, history1 = AdProxGrad(
        oracle_f,
        g,
        prox_g,
        y0;
        maxit = N,
        tol = tol,
        stop = "res",
        lns_init = false,
        verbose = true,
        ver = 2,
        track = ["res", "obj", "grad", "steps"],
        fixed_step = 1e-3,
    )

    y3, history3 = ProxGrad(
        oracle_f,
        g,
        prox_g,
        y0;
        maxit = N,
        tol = tol,
        stop = "res",
        lns_init = false,
        lns = true,
        verbose = true,
        track = ["res", "obj", "grad", "steps"],
        fixed_step = 1e-3,
        inc = 1.1,
        dcr = 0.5,
    )

end


incr = [1.1, 1.2, 1.5]
dcr = [0.5, 0.8, 0.9]
lns_param = Iterators.product(incr, dcr) |> collect
prox_grad_results = Dict()

if RUN_lns && ~LOAD
    @time y1, history1 = AdProxGrad(
        oracle_f,
        g,
        prox_g,
        y0;
        maxit = N,
        tol = tol,
        stop = "res",
        lns_init = false,
        verbose = true,
        ver = 2,
        track = ["res", "obj", "grad", "steps"],
        fixed_step = 1e-3,
    )

    for (a, b) in [lns_param...]
        @time y3, history3, total = ProxGrad(
            oracle_f,
            g,
            prox_g,
            y0;
            maxit = N,
            tol = tol,
            stop = "res",
            lns_init = false,
            lns = true,
            verbose = true,
            track = ["res", "obj", "grad", "steps"],
            fixed_step = 1e-3,
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
