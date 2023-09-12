using LinearAlgebra
using ForwardDiff: gradient
using Random
include("misc.jl")
include("alg.jl")
include("plotting.jl")



# Problem source: Minimum length piecewise-linear curve subject to equality constraints. We solve primal problem via projection onto affine subspace. Example 10.4 in B & V
name = "min_len_curve"


# gradient is cheap, so we only compute projections
function cost_lns(niter, nlns)
    N_total = nlns
    return N_total
end

RUN = true
RUN_lns = false
SAVE = false
LOAD = false
PLOTS = false
Random.seed!(2023)

### Defining a problem
m, n = 50, 500
name_instance = name * "_m=$(m)_n=$(n)"
x_feas = randn(n)
A = randn(m, n)
b = A * x_feas

P = inv(A * A')


f(x) = sqrt(1 + x[1]^2) + sum(sqrt.(1 .+ (x[2:end] - x[1:end-1]) .^ 2))

df(x) = gradient(f, x)



function oracle_f(x)
    return f(x), df(x)
end



g(x) = 0.0
prox_g(x, α) = x - A' * (P * (A * x - b))


x0 = prox_g(rand(n), 1)
tol = 1e-6
N = 1700

if RUN
    x1, history1 = AdProxGrad(
        oracle_f,
        g,
        prox_g,
        x0;
        maxit = N,
        tol = tol,
        stop = "res",
        lns_init = true,
        verbose = true,
        ver = 2,
        track = ["res", "obj", "grad", "steps"],
        fixed_step = 0.001,
    )

    x3, history3 = ProxGrad(
        oracle_f,
        g,
        prox_g,
        x0;
        maxit = N,
        tol = tol,
        stop = "res",
        lns_init = true,
        lns = true,
        verbose = true,
        track = ["res", "obj", "grad", "steps"],
        fixed_step = 0.001,
        inc = 1.1,
        dcr = 0.5,
    )

end


incr = [1.1, 1.2, 1.5]
dcr = [0.5, 0.8, 0.9]
lns_param = Iterators.product(incr, dcr) |> collect
prox_grad_results = Dict()

if RUN_lns && ~LOAD
    @time x1, history1 = AdProxGrad(
        oracle_f,
        g,
        prox_g,
        x0;
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
        @time x3, history3, total = ProxGrad(
            oracle_f,
            g,
            prox_g,
            x0;
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
