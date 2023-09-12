using LinearAlgebra



function oracle!(x, fx, grad_fx)
    return fx, grad_fx
end


function stop(it, err, maxit, tol)
    if it > maxit || err <= tol
        return false
    end
end


"""
linesearch_initial: Initial linesearch in order to find α_0. Contains two loops to ensure that
the stepsize is neither too small, nor too large.
"""
function linesearch_initial(oracle_f, g, prox_g, x0, α)
    obj_x, grad_fx0 = oracle_f(x0)
    largestep = true
    for i in range(1, 100)
        x1 = prox_g(x0 - α * grad_fx0, α)
        obj_x1, grad_fx1 = oracle_f(x1)
        if i == 1 && isapprox(x0, x1)
            println("Congrats: initial x0 is a solution")
        else
            L = norm(grad_fx1 - grad_fx0) / norm(x1 - x0)
            if α * L > 2
                # decrease step
                largestep = false
                println("/2")
                α *= 0.5
            else
                if α * L <= 2 && largestep
                    # increase step aggressively
                    α *= 10
                    println("*10")
                    if α > 10
                        return α, x1, grad_fx0, grad_fx1, obj_x, obj_x1, i
                        break
                    end

                else
                    return α, x1, grad_fx0, grad_fx1, obj_x, obj_x1, i
                    break
                end
            end
        end
    end

end

"""
linesearch: Main linesearch in order to find α_k in every iteration. It is based on Armijo rule.
In the beginning we increase stepsize `inc` times and then decrease it with a factor `dcr` until the condition is met
"""
function linesearch(oracle_f, prox_g, x, fx, grad_fx, α, counter; inc = 1.2, dcr = 0.5)
    α *= inc
    #fx, grad_fx = oracle_f(x)
    for i in range(1, 1000)
        counter += 1
        x1 = prox_g(x - α * grad_fx, α)
        fx1, grad_fx1 = oracle_f(x1)
        if fx1 <= fx + dot(grad_fx, x1 - x) + 1.0 / (2 * α) * norm(x1 - x)^2
            x, fx, grad_fx = x1, fx1, grad_fx1
            break
        else
            α *= dcr
        end
    end
    return x, fx, grad_fx, α, counter
end


"""
collect_history: is used to accumulate all the important information during iterations.
"""
function collect_history(history_dic, data_dic)
    for key in keys(history_dic)
        push!(history_dic[key], data_dic[key])
    end
    return history_dic
end

"""
AdProxGrad: main framework for Proximal Gradient method
ver = 1: Algorithm 1 + Prox as in the paper
ver = 2: Algorithm 3 = Algorithm 2 + Prox
ver = 3: GD with a fixed step
ver = 4: GD with linesearch (basic one). More advances is impemented in `ProxGrad`. 
"""
function AdProxGrad(
    oracle_f,
    g,
    prox_g,
    x0;
    maxit = 1000,
    tol = 1e-9,
    stop = "res",
    lns_init = true,
    verbose = false,
    ver = 2,
    track = ["res", "obj", "grad", "steps"],
    fixed_step = 1e-2,
)
    obj(fx, x) = fx + g(x)
    x_prev = x0
    θ = 1.0 / 3
    track_steps = Int8[]
    if lns_init
        α_prev, x, grad_prev, grad_fx, f_prev, fx, lns_iter =
            linesearch_initial(oracle_f, g, prox_g, x0, fixed_step)
        verbose &&
            println("Linesearch found initial stepsize $α_prev in $lns_iter iterations")
    else
        α_prev = fixed_step
        verbose && println("No linesearch, initial stepsize is ", α_prev)
        f_prev, grad_prev = oracle_f(x_prev)
        x = prox_g(x_prev - α_prev * grad_prev, α_prev)
        fx, grad_fx = oracle_f(x)
    end
    dict = Dict(
        "res" => [norm(x - x_prev) / α_prev],
        "obj" => [obj(f_prev, x_prev)],
        "grad" => norm.([grad_prev]),
        "steps" => [α_prev],
    )
    history_dic = filter(p -> p.first in track, dict)
    i = 1
    for i in range(1, maxit)
        if ver == 1
            L = norm(grad_fx - grad_prev) / norm(x - x_prev)
            α = min(sqrt(1 + θ) * α_prev, 1 / (sqrt(2) * L))
        elseif ver == 2
            L = norm(grad_fx - grad_prev) / norm(x - x_prev)
            α = min(sqrt(2 / 3 + θ) * α_prev, α_prev / sqrt(max(2 * α_prev^2 * L^2 - 1, 0)))
            opt = sqrt(2 / 3 + θ) < 1.0 / sqrt(max(2 * α_prev^2 * L^2 - 1, 0)) ? 1 : 2
            append!(track_steps, opt)
        elseif ver == 3
            α = fixed_step
        end

        θ = α / α_prev
        x_prev, grad_prev, α_prev = x, grad_fx, α
        x = prox_g(x - α * grad_fx, α)


        residual = norm(x_prev - x) / α
        current_info = Dict(zip(track, [residual, obj(fx, x), norm(grad_fx), α]))
        if ver != 4
            fx, grad_fx = oracle_f(x)
        end

        collect_history(history_dic, current_info)
        if current_info[stop] <= tol
            verbose && println("The algorithm reached required accuracy in $i iterations")
            break
        end

    end

    return x, history_dic, track_steps
end


function ProxGrad(
    oracle_f,
    g,
    prox_g,
    x0;
    maxit = 1000,
    tol = 1e-9,
    stop = "res",
    lns_init = true,
    lns = true,
    verbose = false,
    track = ["res", "obj", "grad", "steps"],
    fixed_step = 1e-2,
    inc = 1.2,
    dcr = 0.5,
)
    obj(fx, x) = fx + g(x)
    total = 0
    x_prev = x0
    if lns_init
        α_prev, x, grad_prev, grad_fx, f_prev, fx, lns_iter =
            linesearch_initial(oracle_f, g, prox_g, x0, fixed_step)
        verbose &&
            println("Linesearch found initial stepsize $α_prev in $lns_iter iterations")
    else
        α_prev = fixed_step
        verbose && println("No linesearch, initial stepsize is ", α_prev)
        f_prev, grad_prev = oracle_f(x_prev)
        x = prox_g(x_prev - α_prev * grad_prev, α_prev)
        fx, grad_fx = oracle_f(x)
    end
    dict = Dict(
        "res" => [norm(x - x_prev) / α_prev],
        "obj" => [obj(f_prev, x_prev)],
        "grad" => ([norm(grad_prev)]),
        "steps" => [α_prev],
    )
    history_dic = filter(p -> p.first in track, dict)

    i = 0
    for _ in range(1, maxit)
        i += 1
        if lns
            x, fx, grad_fx, α, total = linesearch(
                oracle_f,
                prox_g,
                x,
                fx,
                grad_fx,
                α_prev,
                total;
                inc = inc,
                dcr = dcr,
            )
        else
            α = fixed_step
            x = prox_g(x - α * grad_fx, α)
            fx, grad_fx = oracle_f(x)
            total += 1
        end

        residual = norm(x_prev - x) / α
        x_prev, α_prev = x, α
        current_info = Dict(zip(track, [residual, obj(fx, x), norm(grad_fx), α]))


        collect_history(history_dic, current_info)
        if current_info[stop] <= tol
            verbose && println("The algorithm reached required accuracy in $i iterations.")
            break
        end

    end
    avg_calls = round(total / i, digits = 1)
    verbose && println(
        "The total number of oracle calls is $(total). In average per iteration, it required $avg_calls calls",
    )
    return x, history_dic, total
end




# function update_iter!(oracle_f, prox_g, x, x_prev, grad, grad_prev)

#     L = norm(grad - grad_prev) / norm(x - x_prev)
#     if ver == 1
#         α = min(sqrt(1 + θ) * α_prev, 1 / (sqrt(2) * L))
#     elseif ver == 2
#         α = min(sqrt(2 / 3 + θ) * α_prev, α_prev / sqrt(max(2 * α_prev^2 * L^2 - 1, 0)))
#     elseif ver == 0
#         α = fixed_step
#     end
#     θ = α / α_prev
#     x_prev, grad_prev, α_prev = x, grad_x, α
#     x = prox_g(x - α * grad, α)
#     obj_x, grad = oracle_f(x)
#     ### smth with history
#     return x, x_prev, grad, grad_prev
# end
