using Flux
using NonlinearSolve, MINPACK
using Plots

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
output_dims = 3
num_layers = 3
num_neurals = 10
layers_neurals = fill(num_neurals, num_layers + 1)
model = f64(create_network(input_dims, output_dims, layers_neurals))
println(model)
# 提取模型参数
flat, rebuild = Flux.destructure(model)

# 参数设定
rho = 1.0 # 密度
Re = 100.0 # 雷诺数
v_slip = 1.0 # 滑移速度
De = 1.0 # 水力直径
mu = rho*v_slip*De/Re # 动力粘度
params = (rho, Re, v_slip, De)
# 网格设定
Lx = 0.0
Rx = 1.0
nx = 15
dx = (Rx - Lx)/(nx-1)
x = range(Lx, Rx, length=nx)
Ly = 0.0
Ry = 1.0
ny = 15
dy = (Ry - Ly)/(ny-1)
y = range(Ly, Ry, length=ny)
# u 边界条件
Loss_bc_u_down = zeros(nx)  # 下
Loss_bc_u_up = zeros(nx)    # 上
Loss_bc_u_left = zeros(ny)  # 左
Loss_bc_u_right = zeros(ny) # 右
# v 边界条件
Loss_bc_v_down = zeros(nx)  # 下
Loss_bc_v_up = zeros(nx)    # 上
Loss_bc_v_left = zeros(ny)  # 左
Loss_bc_v_right = zeros(ny) # 右
# p 边界条件
Loss_bc_p_down = zeros(nx)  # 下
Loss_bc_p_up = zeros(nx)    # 上
Loss_bc_p_left = zeros(ny)  # 左
Loss_bc_p_right = zeros(ny) # 右
Loss_bc_p_points = zeros(1)
# 内部条件
Loss_mass = zeros(nx, ny)
Loss_momentx = zeros(nx, ny)
Loss_momenty = zeros(nx, ny)
# 迭代记号
iterate = [0]

# 定义jacobi迭代函数
function Fsolve(init_params,params)
    rho = params[1]
    Re = params[2]
    v_slip = params[3]
    De = params[4]
    mu = rho*v_slip*De/Re
    #println(maximum(init_params))
    model = f64(rebuild(init_params))
    # 定义神经网络函数
    u(x,y) = model([x,y])[1]
    # 定义u对x的一次导数
    u_x(x,y) = (u(x+dx,y) - u(x-dx,y))/(2*dx)
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
    # 定义u对y的一次导数
    u_y(x,y) = (u(x,y+dy) - u(x,y-dy))/(2*dy)
    # 定义u对y的迎风导数
    function u_y_upwind(x,y,f) 
        if f > 0
            return (u(x,y) - u(x,y - dy))/dy
        elseif f < 0
            return (u(x,y + dy) - u(x,y))/dy
        else
            return (u(x,y + dy) - u(x,y - dy))/(2*dy)
        end
    end
    # 定义u对y的二次导数
    u_yy(x,y) = (u(x,y+dy) - 2*u(x,y) + u(x,y-dy))/(dy^2)
    # 定义神经网络函数
    v(x,y) = model([x,y])[2]
    # 定义v对x的一次导数
    v_x(x,y) = (v(x+dx,y) - v(x-dx,y))/(2*dx)
    # 定义v对x的迎风导数
    function v_x_upwind(x,y,f) 
        if f > 0
            return (v(x,y) - v(x - dx,y))/dx
        elseif f < 0
            return (v(x + dx,y) - v(x,y))/dx
        else
            return (v(x + dx,y) - v(x - dx,y))/(2*dx)
        end
    end
    # 定义v对x的二次导数
    v_xx(x,y) = (v(x+dx,y) - 2*v(x,y) + v(x-dx,y))/(dx^2)
    # 定义v对y的一次导数
    v_y(x,y) = (v(x,y+dy) - v(x,y-dy))/(2*dy)
    # 定义v对y的迎风导数
    function v_y_upwind(x,y,f) 
        if f > 0
            return (v(x,y) - v(x,y - dy))/dy
        elseif f < 0
            return (v(x,y + dy) - v(x,y))/dy
        else
            return (v(x,y + dy) - v(x,y - dy))/(2*dy)
        end
    end
    # 定义v对y的二次导数
    v_yy(x,y) = (v(x,y+dy) - 2*v(x,y) + v(x,y-dy))/(dy^2)
    # 定义神经网络函数
    p(x,y) = model([x,y])[3]
    # 定义p对x的一次导数
    p_x(x,y) = (p(x+dx,y) - p(x-dx,y))/(2*dx)
    # 定义p对y的一次导数
    p_y(x,y) = (p(x,y+dy) - p(x,y-dy))/(2*dy)

    # 计算压力点约束
    Loss_bc_p_points[1] = p(x[nx], y[ny]) - 0.0
    # 计算下边界条件 - 虚单元
    Threads.@threads for i in eachindex(x)
        Loss_bc_u_down[i] = u(x[i], y[1] - dy) + u(x[i], y[1] + dy)
        Loss_bc_v_down[i] = v(x[i], y[1] - dy) + v(x[i], y[1] + dy)
        Loss_bc_p_down[i] = p_y(x[i], y[1] - dy) + p_y(x[i], y[1] + dy)
    end
    # 计算上边界条件 - 虚单元
    Threads.@threads for i in eachindex(x)
        Loss_bc_u_up[i] = u(x[i], y[ny] - dy) + u(x[i], y[ny] + dy) - 2*v_slip
        Loss_bc_v_up[i] = v(x[i], y[ny] - dy) + v(x[i], y[ny] + dy)
        Loss_bc_p_up[i] = p_y(x[i], y[ny] - dy) + p_y(x[i], y[ny] + dy)
    end
    # 计算左边界条件 - 虚单元
    Threads.@threads for i in eachindex(y)
        Loss_bc_u_left[i] = u(x[1] - dx, y[i]) + u(x[1] + dx, y[i])
        Loss_bc_v_left[i] = v(x[1] - dx, y[i]) + v(x[1] + dx, y[i])
        Loss_bc_p_left[i] = p_x(x[1] - dx, y[i]) + p_x(x[1] + dx, y[i])
    end
    # 计算右边界条件 - 虚单元
    Threads.@threads for i in eachindex(y)
        Loss_bc_u_right[i] = u(x[nx] - dx, y[i]) + u(x[nx] + dx, y[i]) 
        Loss_bc_v_right[i] = v(x[nx] - dx, y[i]) + v(x[nx] + dx, y[i])
        Loss_bc_p_right[i] = p_x(x[nx] - dx, y[i]) + p_x(x[nx] + dx, y[i])
    end
    # 计算内部NS方程约束
    @sync begin
        for i in eachindex(x)
            @async begin
                Threads.@threads for j in eachindex(y)
                    U = u(x[i], y[j])
                    V = v(x[i], y[j])

                    Ux = u_x(x[i], y[j])
                    Ux_upwind = u_x_upwind(x[i], y[j], U)
                    Uy_upwind = u_y_upwind(x[i], y[j], V)
                    Uxx = u_xx(x[i], y[j])
                    Uyy = u_yy(x[i], y[j])

                    Vy = v_y(x[i], y[j])
                    Vx_upwind = v_x_upwind(x[i], y[j], U)
                    Vy_upwind = v_y_upwind(x[i], y[j], V)
                    Vxx = v_xx(x[i], y[j])
                    Vyy = v_yy(x[i], y[j])

                    Px = p_x(x[i], y[j])
                    Py = p_y(x[i], y[j])

                    Loss_mass[i, j] = Ux + Vy
                    Loss_momentx[i, j] = rho*(U*Ux_upwind + V*Uy_upwind) + Px - mu*(Uxx + Uyy)
                    Loss_momenty[i, j] = rho*(U*Vx_upwind + V*Vy_upwind) + Py - mu*(Vxx + Vyy)
                end
            end
        end
    end
    LossBond = cat(Loss_bc_p_points,
                   Loss_bc_u_down, 
                   Loss_bc_u_up, 
                   Loss_bc_u_left, 
                   Loss_bc_u_right, 
                   Loss_bc_v_down, 
                   Loss_bc_v_up, 
                   Loss_bc_v_left, 
                   Loss_bc_v_right, 
                   Loss_bc_p_down, 
                   Loss_bc_p_up, 
                   Loss_bc_p_left, 
                   Loss_bc_p_right,
                   dims = 1)
    LossInter = cat(vec(Loss_mass),
                    vec(Loss_momentx), 
                    vec(Loss_momenty), 
                    dims = 1)
    Vecs = cat(LossBond, LossInter, dims = 1)
    iterate[1] = iterate[1] + 1
    if iterate[1] % 500 == 0 || iterate[1] == 1
        println("iterate: ", iterate[1], "\tL2 loss: ", (sum(Vecs.^2)/length(Vecs))^0.5)
    end
    return Vecs
end

prob = NonlinearProblem(Fsolve, flat, params)
@time sol = solve(prob, CMINPACK(method = :lmdif))

model = rebuild(sol.u)

plot_nx = 100
plot_ny = 100
plot_x = range(Lx, Rx, length=plot_nx)
plot_y = range(Ly, Ry, length=plot_ny)
u_num = ones(plot_nx, plot_ny)
v_num = ones(plot_nx, plot_ny)
p_num = ones(plot_nx, plot_ny)
for i in eachindex(plot_x)
    for j in eachindex(plot_y)
        u_num[i, j] = model([plot_x[i], plot_y[j]])[1]
        v_num[i, j] = model([plot_x[i], plot_y[j]])[2]
        p_num[i, j] = model([plot_x[i], plot_y[j]])[3]
    end
end
u_num = u_num'
v_num = v_num'
p_num = p_num'
p1 = plot(plot_x, plot_y, u_num, linetype =:contourf)
p2 = plot(plot_x, plot_y, v_num, linetype =:contourf)
p3 = plot(plot_x, plot_y, p_num, linetype =:contourf)
plot(p1, p2, p3, size = (800,600))
