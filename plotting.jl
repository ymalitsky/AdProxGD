using Plots
using LinearAlgebra
using LaTeXStrings
default(palette = palette(:mk_12))
plot_font = "Computer Modern"
default(
    fontfamily = plot_font,
    linewidth = 4,
    framestyle = :box,
    markersize = 11,
    #label=nothing,
    grid = true,
    #tickfontsize=7,
    xtickfontsize = 15,
    ytickfontsize = 15,
    guidefontsize = 20,
    legendfontsize = 12,
    titlefontsize = 15,
    right_margin = 5 * Plots.mm,
)

"""
Find the smallest objective value across all the algorithms
"""

function find_min_value(ad_prox_results, prox_grad_results)
    f_min = minimum(ad_prox_results["obj"])
    for k in keys(prox_grad_results)
        values = prox_grad_results[k][1]["obj"]
        f_min2 = minimum(values)
        f_min = min(f_min, f_min2)
    end
    return f_min
end

function plot_res(
    ad_prox_results,
    prox_grad_results,
    key,
    name,
    cost_lns;
    total_cost = false,
)
    # if we plot the objective, we should pot f(xk)-f_*
    if key == "obj"
        opt = find_min_value(ad_prox_results, prox_grad_results)
    else
        opt = 0.0
    end
    adpg = ad_prox_results[key] .- opt
    p = plot(adpg[adpg.>0], yscale = :log10, label = "AdProxGD", color = :black)
    for k in keys(prox_grad_results)
        values = prox_grad_results[k][1][key] .- opt
        N = length(values)
        N_lns = prox_grad_results[k][2]
        N_total = cost_lns(N, N_lns)
        values = values[values.>0]
        N = length(values)
        if total_cost
            plot!(p, range(1, N_total, N), values, label = "$k")
        else
            plot!(p, values, label = "$k")
        end
    end
    savefig(p, "./plots/$(name).pdf")
end

function plot_res_for_paper(name, key, cost_lns, uplim, xl, yl, spr_coef)
    # load data
    ad_prox_results, prox_grad_results = load_data(name)
    # if we plot the objective, we should pot f(xk)-f_*
    if key == "obj"
        opt = find_min_value(ad_prox_results, prox_grad_results)
    else
        opt = 0.0
    end
    adpg = ad_prox_results[key] .- opt
    adpg_values = adpg[adpg.>0][1:min(end, uplim)]
    adpg_values = sparsify_results(adpg_values, spr_coef)
    p = plot(
        adpg_values,
        yscale = :log10,
        label = "AdProxGD",
        color = :black,
        xlabel = xl,
        ylabel = yl,
    )
    for k in keys(prox_grad_results)
        values = prox_grad_results[k][1][key] .- opt
        N = length(values)
        N_lns = prox_grad_results[k][2]
        N_total = cost_lns(N, N_lns)
        values = values[values.>0]
        N = length(values)
        xvalues, yvalues = apply_bound(range(1, N_total, N), values, uplim)
        lns_values = sparsify_results(xvalues, yvalues, spr_coef)
        plot!(p, lns_values, label = "$k")
    end
    p_nolegend = plot(p, legend = false)
    savefig(p, "./plots/$(name)_$(key)_paper.pdf")
    savefig(p_nolegend, "./plots/$(name)_$(key)_paper_no_legend.pdf")
end


function quick_plot(history1, history2, key)
    data1 = history1[key]
    data2 = history2[key]
    p1 = plot(data1[data1.!=0], yscale = :log10, label = "AdProxGD")
    plot!(p1, data2[data2.!=0], label = "ProxGD-lns")
end


function sparsify_results(xs, ys, M)
    return collect(zip(xs, ys))[1:M:end]
end

function sparsify_results(ys, M)
    N = length(ys)
    return collect(zip(range(1, N), ys))[1:M:end]
end


function export_palette_tikz(N)
    cols = (ColorSchemes.mk_12)[1:N]
    return cols * 255
end
