using LinearAlgebra
using JLD2, FileIO
using DataStructures: MutableBinaryMaxHeap, top_with_handle, delete!


""" Algorithm from van den Berg, E., Friedlander, M.P.: 'Probing the
Pareto frontier for basis pursuit solution'. The description is from
Condat L: 'Fast Projection onto the Simplex and the l1
Ball'. (Algorithm 2)"""

function proj_simplex(y::Array{T,1}, a::T = one(T)) where {T<:Real}
    N = length(y)
    if sum(y) == a && all(y .≥ 0)
        x = y
    else
        τ = zero(T)
        v = MutableBinaryMaxHeap(y)
        cumsum_u = zero(T)
        for k = 1:N
            u = first(v)
            if cumsum_u + u < k * u + a
                cumsum_u += u
                i = top_with_handle(v)[2]
                delete!(v, i)
                τ = (cumsum_u - a) / k
            else
                break
            end
        end
        x = max.(y .- τ, zero(T))
    end
    return x
end

function save_data(history1, prox_grad_results, name)
    file_name = "./saved_data/" * name * ".jld2"
    data = Dict("AdProxGD" => history1, "ProxGD" => prox_grad_results)
    save(file_name, data)
end

function load_data(name)
    file_name = "./saved_data/" * name * ".jld2"
    data = load(file_name)
    history1 = data["AdProxGD"]
    prox_grad_results = data["ProxGD"]
    return history1, prox_grad_results
end

function make_all_plots(history1, prox_grad_results, name_instance)
    name_total = name_instance * "_total"
    name_obj_total = name_instance * "_obj_total"
    name_obj = name_instance * "_obj"
    plot_res(history1, prox_grad_results, "res", name_total, cost_lns, total_cost = true)
    plot_res(
        history1,
        prox_grad_results,
        "res",
        name_instance,
        cost_lns,
        total_cost = false,
    )
    plot_res(
        history1,
        prox_grad_results,
        "obj",
        name_obj_total,
        cost_lns,
        total_cost = true,
    )
    plot_res(history1, prox_grad_results, "obj", name_obj, cost_lns, total_cost = false)
end

"""
Apply an upper bound for x axis and do the same respectively for y axis
"""
function apply_bound(xs, ys, up_lim)
    xs = xs[xs.<=up_lim]
    l = length(xs)
    ys = ys[1:l]
    return xs, ys
end
