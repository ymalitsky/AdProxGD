using LinearAlgebra
using Random
include("misc.jl")
include("alg.jl")
include("plotting.jl")



# Problem source:
# F(X) = 1/2 |P_Ω (X - A)|^2 s.t. |X|_* ≤ r
name = "low_rank_matr_compl"
Random.seed!(2023)

# The expensive operation here is svd which is needed to compute for prox
function cost_lns(niter, nlns)
    N_total = nlns
    return N_total
end


RUN = true
RUN_lns = false
SAVE = false
LOAD = false
PLOTS = false

### Defining a problem


m, r, n = 200, 20, 200
name_instance = name * "_m=$(m)_n=$(n)_r=$r"


A = randn(m, r) * randn(r, n)
observed = 0.2
mask = unique(sort(rand(1:m*n, Int(ceil(observed * m * n)))))


function f(X)
    return 0.5 * norm(X[mask] - A[mask])^2
end


function oracle_f(X)
    res = X[mask] - A[mask]
    G = zeros(m, n)
    G[mask] = res
    return 0.5 * norm(res)^2, G
end

g(X) = 0.0
function prox_g(X, α)
    U, sigma, Vt = svd(X)
    sigma[r+1:end] .= 0
    return U * (sigma .* Vt')
end




X0 = randn(m, n)
X0[mask] .= A[mask]

tol = 1e-6
N = 20000

if RUN
    X1, history2 = AdProxGrad(
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
        fixed_step = 1.0,
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
        lns = false,
        verbose = true,
        track = ["res", "obj", "grad", "steps"],
        fixed_step = 1.0,
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
