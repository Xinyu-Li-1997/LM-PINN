using Flux
using Zygote
using NonlinearSolve, MINPACK
using Plots, CSV, DataFrames
#using Enzyme, ForwardDiff
#using Optimization, OptimizationOptimJL
#using LeastSquaresOptim, LinearSolve, Sundials, NLsolve

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

# 创建模型实例
input_dims = 2
output_dims = 1
num_layers = 2
num_neurals = 8
maxiters = 100000 # 预处理最大训练步数
layers_neurals = fill(num_neurals, num_layers + 1)
model = f64(create_network(input_dims, output_dims, layers_neurals))
# 提取模型参数
flat, rebuild = Flux.destructure(model)
#flat = flat./(flat.+ 1e-6).- 1.0

# 参数设定
alpha = -0.0
# 网格设定
Lx = -1.0
Rx = 1.0
nx = 15
dx = (Rx - Lx)/(nx-1)
x = range(Lx, Rx, length=nx)
Ly = 0.0
Ry = 1.0
ny = 15
dy = (Ry - Ly)/(ny-1)
y = range(Ly, Ry, length=ny)
Loss = ones(nx, ny)
iterate = [0]
Error = []
Iters = []
# 定义jacobi迭代函数
function Fsolve(init_params,p)
    #println(maximum(init_params))
    model = f64(rebuild(init_params))
    # 定义神经网络函数
    u(x,y) = model([x,y])[1]
    # 定义u对x的迎风导数
    function u_x_upwind(x,y,f) 
        if f > 0
            return (u(x,y) - u(x - dx,y))/dx
        elseif f < 0
            return (u(x + dx,y) - u(x,y))/dx
        else
            return (u(x + dx,y) - u(x - dx,y))/(2*dx)
        end
    end
    # 定义u对x的二次导数
    u_xx(x,y) = (u(x+dx,y) - 2*u(x,y) + u(x-dx,y))/(dx^2)
    # 定义u对y的一次导数(时域差分)
    u_y(x,y) = (u(x,y + dy) - u(x,y))/dy
    # 计算初始条件
    Threads.@threads for i in eachindex(x)
        Loss[i, 1] = u(x[i], y[1]) + sin.(pi*x[i])
    end
    # 计算左边界条件
    Threads.@threads for i in eachindex(y)
        Loss[1, i] = u(x[1], y[i]) - 0.0
    end
    # 计算右边界条件
    Threads.@threads for i in eachindex(y)
        Loss[nx, i] = u(x[nx], y[i]) - 0.0
    end
    # 计算发展条件
    Threads.@threads for i in eachindex(x)
        Loss[i, ny] = u_y(x[i], y[ny]) + u(x[i], y[ny])*u_x_upwind(x[i], y[ny],u(x[i], y[ny])) + p*u_xx(x[i], y[ny])
    end
    # 计算内部非线性方程约束
    Threads.@threads for i in 2:nx-1
        Threads.@threads for j in 2:ny-1
            Loss[i, j] = u_y(x[i], y[j]) + u(x[i], y[j])*u_x_upwind(x[i], y[j],u(x[i], y[j])) + p*u_xx(x[i], y[j])
        end
    end
    # 返回展平后的Loss以便于优化器处理
    Vecs = vec(Loss)
    L2 = (sum(Vecs.^2)/length(Vecs))^0.5
    iterate[1] = iterate[1] + 1
    if iterate[1] % 500 == 0 || iterate[1] == 1
        println("iterate: ", iterate[1], "\tL2 loss: ", L2)
        push!(Iters, iterate[1])
        push!(Error, L2)
    end
    # 设置终止条件
    if iterate[1] < maxiters
        return Vecs
    else
        return zeros(length(Vecs))
    end
end

function Fsolve_optim(init_params,p)
    VEC = Fsolve(init_params,p)
    return sum(VEC.^2)/length(VEC)
end
# minpack - OK
prob = NonlinearProblem(Fsolve, flat, alpha)
@time sol = solve(prob, CMINPACK(method = :lm))
model = rebuild(sol.u)

plot_nx = 100
plot_ny = 100
plot_x = range(Lx, Rx, length=plot_nx)
plot_y = range(Ly, Ry, length=plot_ny)
u_num = ones(plot_ny, plot_nx)
#u_ana = ones(plot_nx, plot_ny)
for i in eachindex(plot_x)
    for j in eachindex(plot_y)
        u_num[i, j] = model([plot_x[i], plot_y[j]])[1]
        #u_ana[i, j] = u_anal(plot_x[i], plot_y[j])
    end
end

p1 = plot(plot_y, plot_x, u_num, linetype =:contourf)
pp = plot(p1, size = (800,600))
display(pp)
df = DataFrame(Iters=Iters, Error=Error)
CSV.write("R_IM_FD_α_$(alpha).csv", DataFrame(df))