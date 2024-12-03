using Flux
using Zygote
using NonlinearSolve, Optimization
using Plots, CSV, DataFrames

# 定义模型
function create_network(input_dim::Int, output_dim::Int, hidden_layers::Vector{Int})
    layers = []
    last_dim = input_dim
    for layer_dim in hidden_layers
        push!(layers, Dense(last_dim => layer_dim, gelu))
        last_dim = layer_dim
    end
    # 添加输出层，没有激活函数
    push!(layers, Dense(last_dim => output_dim))
    model = Chain(layers...)
    return model
end

nx = 21
ny = 21
num_layers = 2
num_neurals = 13
Flags = "AD"

# 创建模型实例
input_dims = 2
output_dims = 1
layers_neurals = fill(num_neurals, num_layers + 1)
model = f64(create_network(input_dims, output_dims, layers_neurals))
# 提取模型参数
flat, rebuild = Flux.destructure(model)

# 网格设定
Lx = 0.0
Rx = 1.0
dx = (Rx - Lx)/(nx-1)
x = range(Lx, Rx, length=nx)
Ly = 0.0
Ry = 1.0
dy = (Ry - Ly)/(ny-1)
y = range(Ly, Ry, length=ny)
Loss = ones(nx, ny)
iterate = [0]
maxiters_initial = 2000000 # 预处理最大训练步数

# 定义jacobi迭代函数
function Fsolve_AD(init_params,p)
    #println(maximum(init_params))
    model = f64(rebuild(init_params))
    # 定义神经网络函数
    u(x,y) = model([x,y])[1]
    # 预先定义u对x的一次导数
    #u_xx(x,y) = (u(x+dx,y) - 2*u(x,y) + u(x-dx,y))/(dx^2)
    #u_yy(x,y) = (u(x,y+dy) - 2*u(x,y) + u(x,y-dy))/(dy^2)
    u_xx(x,y) = Zygote.hessian(x->u(x,y)[1], x)[1]
    u_yy(x,y) = Zygote.hessian(y->u(x,y)[1], y)[1]
    # 计算下边界条件
    Threads.@threads for i in eachindex(x)
        Loss[i, 1] = u(x[i], y[1]) - 0.0
    end
    # 计算左边界条件
    Threads.@threads for i in eachindex(y)
        Loss[1, i] = u(x[1], y[i]) - 0.0
    end
    # 计算右边界条件
    Threads.@threads for i in eachindex(y)
        Loss[nx, i] = u(x[nx], y[i]) - 0.0
    end
    # 计算上边界条件
    Threads.@threads for i in eachindex(x)
        Loss[i, ny] = u(x[i], y[ny]) - 0.0
    end
    # 计算内部拉普拉斯方程约束
    @sync begin
        for i in 2:nx-1
            @async begin
                Threads.@threads for j in 2:ny-1
                    Loss[i, j] = u_xx(x[i], y[j]) + u_yy(x[i], y[j]) + sin(pi*x[i])*sin(pi*y[j])
                end
            end
        end
    end
    # 返回展平后的Loss以便于优化器处理
    Vecs = vec(Loss)
    iterate[1] = iterate[1] + 1
    L2 = (sum(Vecs.^2)/length(Vecs))^0.5
    if iterate[1] % 500 == 0 || iterate[1] == 1
        println("iterate: ", iterate[1], "\tL2 loss: ", L2)
    end
    return L2 
end

function Fsolve_ND(init_params,p)
    #println(maximum(init_params))
    model = f64(rebuild(init_params))
    # 定义神经网络函数
    u(x,y) = model([x,y])[1]
    # 预先定义u对x的一次导数
    u_xx(x,y) = (u(x+dx,y) - 2*u(x,y) + u(x-dx,y))/(dx^2)
    u_yy(x,y) = (u(x,y+dy) - 2*u(x,y) + u(x,y-dy))/(dy^2)
    #u_xx(x,y) = Zygote.hessian(x->u(x,y)[1], x)[1]
    #u_yy(x,y) = Zygote.hessian(y->u(x,y)[1], y)[1]
    # 计算下边界条件
    Threads.@threads for i in eachindex(x)
        Loss[i, 1] = u(x[i], y[1]) - 0.0
    end
    # 计算左边界条件
    Threads.@threads for i in eachindex(y)
        Loss[1, i] = u(x[1], y[i]) - 0.0
    end
    # 计算右边界条件
    Threads.@threads for i in eachindex(y)
        Loss[nx, i] = u(x[nx], y[i]) - 0.0
    end
    # 计算上边界条件
    Threads.@threads for i in eachindex(x)
        Loss[i, ny] = u(x[i], y[ny]) - 0.0
    end
    # 计算内部拉普拉斯方程约束
    @sync begin
        for i in 2:nx-1
            @async begin
                Threads.@threads for j in 2:ny-1
                    Loss[i, j] = u_xx(x[i], y[j]) + u_yy(x[i], y[j]) + sin(pi*x[i])*sin(pi*y[j])
                end
            end
        end
    end
    # 返回展平后的Loss以便于优化器处理
    Vecs = vec(Loss)
    iterate[1] = iterate[1] + 1
    L2 = (sum(Vecs.^2)/length(Vecs))^0.5
    if iterate[1] % 50000 == 0 || iterate[1] == 1
        println("iterate: ", iterate[1], "\tL2 loss: ", L2)
    end
    return L2
end

if Flags == "ND"
    optf = OptimizationFunction(Fsolve_ND, AutoFiniteDiff())
    prob = Optimization.OptimizationProblem(optf, flat, 0.0)
elseif Flags == "AD"
    optf = OptimizationFunction(Fsolve_AD, AutoFiniteDiff())
    prob = Optimization.OptimizationProblem(optf, flat, 0.0)
else
    error("Flags must be ND or AD")
end

@time sol = solve(prob, Optimization.LBFGS())


#=
model = rebuild(sol.u)
u_anal(x,y) = sin(pi*x)*sin(pi*y)/(2pi^2)

plot_nx = 100
plot_ny = 100
plot_x = range(Lx, Rx, length=plot_nx)
plot_y = range(Ly, Ry, length=plot_ny)
u_num = ones(plot_nx, plot_ny)
u_ana = ones(plot_nx, plot_ny)
for i in eachindex(plot_x)
    for j in eachindex(plot_y)
        u_num[i, j] = model([plot_x[i], plot_y[j]])[1]
        u_ana[i, j] = u_anal(plot_x[i], plot_y[j])
    end
end

diff_u = (u_num - u_ana)
L2_matrix = (sum(diff_u.^2)/(nx*ny))^0.5

p1 = plot(plot_x, plot_y, u_num, linetype =:contourf)
p2 = plot(plot_x, plot_y, u_ana, linetype =:contourf)
p3 = plot(plot_x, plot_y, diff_u, linetype =:contourf)
pp = plot(p1, p2, p3, size = (800,600))
display(pp)
println(L2_matrix)
=#