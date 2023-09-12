using LinearAlgebra
using Random
using ForwardDiff: gradient
include("misc.jl")
include("alg.jl")
include("plotting.jl")



# Problem source: Maximum likelihood estimate of the information matrix
# Equation (7.5) in Boyd
name = "max_loglh"


# Computing prox requires eigendecomposition. And then we can compute cheaply the objective and the next gradient
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
n = 50
lb = 1e-1
kappa = 10000
ub = lb * kappa
M = 100
name_instance = name * "n=$(n)_lb=$(lb)_ub=$(ub)_M=$M"
y = randn(n) * 10

function generate_Y(y, n, M)
    Y = zeros(n, n)
    for i = 1:M
        y_ = y + randn(n)
        Y += y_ * y_'
    end
    return Y / M
end

Y = generate_Y(y, n, M)


f(X) = -logdet(X) + tr(X * Y)

df2(X) = gradient(f, X)
df(X) = -inv(X) + Y


function oracle_f(x)
    return f(x), df(x)
end




g(X) = 0.0
function prox_g(X, α)
    la, U = eigen(Symmetric(X))
    la_ = clamp.(la, lb, ub)
    return U * (la_ .* U')
end



X0 = diagm(lb * ones(n))
tol = 1e-7
N = 10000

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
        fixed_step = 0.001,
    )

    X3, history3 = ProxGrad(
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
