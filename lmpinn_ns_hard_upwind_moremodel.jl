using Flux
using NonlinearSolve, LeastSquaresOptim
using Plots, Printf
using CSV, DataFrames

## 参数设定
input_dims = 2 # x,y
output_dims = 1 # u,v,p
num_layers = 2 # output_dim
num_neurals = 8 # neurals_scale
# 常数
rho = 1.0 # 密度
v_slip = 1.0 # 滑移速度
De = 1.0 # 水力直径
Re = 100.0 # 雷诺数
# 计算域设定
Lx = 0.0
Rx = 1.0
Ly = 0.0
Ry = 1.0
nx = 15
ny = 15
# 计算设定
# 设定雷诺数的变化范围
Re_min = 100.0
Re_max = Re
# 设定步数
N_steps = 0  # 将Re_min到Re_max分成N份（包括两端点）
n_print = 10000 # 每个时间步打印间隔数
maxiters_initial = 1000000 # 预处理最大训练步数
iterate = [0] # 迭代符号

## 定义模型
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
# 创建一个函数来根据长度分割向量
function split_vectors(combined, lengths)
    start = 1
    vectors = []
    for len in lengths
        # 分割出对应的向量
        push!(vectors, combined[start:start+len-1])
        # 更新开始位置
        start += len
    end
    return vectors
end
# 定义差分算子
function fx_up(f, fl, fr, delta) # 一阶迎风差分
    if f > 0.0
        return (f - fl)/delta
    else
        return (fr - f)/delta
    end 
end
function fx(fl, fr, delta) # 一阶中心差分
    return (- fl + fr)/(2*delta) 
end
function fxx(f, fl, fr, delta) # 二阶中心差分
    return (fl - 2*f + fr)/(delta^2)
end
## 创建模型实例
layers_neurals = fill(num_neurals, num_layers + 1)
model_u = f64(create_network(input_dims, output_dims, layers_neurals))
model_v = f64(create_network(input_dims, output_dims, layers_neurals))
model_p = f64(create_network(input_dims, output_dims, layers_neurals))
println("## Finished Initializing the networks ##")
println(model_u)
println(model_v)
println(model_p)
# 提取模型参数
flat_u, rebuild_u = Flux.destructure(model_u)
flat_v, rebuild_v = Flux.destructure(model_v)
flat_p, rebuild_p = Flux.destructure(model_p)
# 获取每个向量的长度
lengths = (length(flat_u), length(flat_v), length(flat_p))
# 将三个向量合并为一个向量
flat = vcat(flat_u, flat_v, flat_p)
N_flat = length(flat)
dxPdy = Int(round(N_flat^0.5))

# 生成参数序列
function generate_paramsinputs()
    paramsinputs = Vector{Tuple}(undef, N_steps + 1)  # 因为我们想要包含最后一个点
    Re_values = range(Re_min, stop=Re_max, length=N_steps + 1)  # 包含两端点
    for i in eachindex(Re_values)
        Re = Re_values[i]
        paramsinput = (rho, Re, v_slip, De, n_print)
        paramsinputs[i] = paramsinput
    end
    return paramsinputs
end
# 输入序列
paramsinputs = generate_paramsinputs()

# 网格设定
#nx = min(nx_max, dxPdy)
dx = (Rx - Lx)/(nx-1)
x = range(Lx, Rx, length=nx)
#ny = min(ny_max, dxPdy)
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
# 数据点空间
up = zeros(nx, ny)
vp = zeros(nx, ny)
pp = zeros(nx, ny)

# 定义稳态Jacobi-Levenberg-Marquardt迭代函数
function Fsolve(init_params,params_in)
    rho = params_in[1]
    Re = params_in[2]
    v_slip = params_in[3]
    De = params_in[4]
    n_print = params_in[5]
    mu = rho*v_slip*De/Re
    # 拆分解向量
    restored_vectors = split_vectors(init_params, lengths)
    # 显式地提取三个向量
    init_params_u = restored_vectors[1]
    init_params_v = restored_vectors[2]
    init_params_p = restored_vectors[3]
    # 正向推理
    model_u = rebuild_u(init_params_u)
    model_v = rebuild_v(init_params_v)
    model_p = rebuild_p(init_params_p)
    # 数据点构建
    Threads.@threads for i in eachindex(x)
        Threads.@threads for j in eachindex(y)
            up[i, j] = model_u([x[i],y[j]])[1]
            vp[i, j] = model_v([x[i],y[j]])[1]
            pp[i, j] = model_p([x[i],y[j]])[1]
        end
    end
    # 计算压力点约束
    Loss_bc_p_points[1] = pp[nx-1, ny-1] - 0.0

    # 边界条件约束
    Threads.@threads for i in eachindex(x)
        @inbounds begin
            # 计算下边界条件 - 硬约束
            Loss_bc_u_down[i] = up[i, 1] - 0.0
            Loss_bc_v_down[i] = vp[i, 1] - 0.0
            Loss_bc_p_down[i] = pp[i, 1] - pp[i, 2] # 零法向梯度
            # 计算上边界条件 - 硬约束
            Loss_bc_u_up[i] = up[i, ny] - v_slip
            Loss_bc_v_up[i] = vp[i, ny] - 0.0
            Loss_bc_p_up[i] = pp[i, ny] - pp[i, ny-1] # 零法向梯度
        end
    end
    Threads.@threads for i in eachindex(y)
        if i == 1 || i == ny # 上下底边防止过约束
            nothing
        else
            @inbounds begin
                # 计算左边界条件 - 硬约束
                Loss_bc_u_left[i] = up[1, i] - 0.0
                Loss_bc_v_left[i] = vp[1, i] - 0.0
                Loss_bc_p_left[i] = pp[1, i] - pp[2, i] # 零法向梯度
                # 计算右边界条件 - 硬约束
                Loss_bc_u_right[i] = up[nx, i] - 0.0
                Loss_bc_v_right[i] = vp[nx, i] - 0.0
                Loss_bc_p_right[i] = pp[nx, i] - pp[nx-1, i] # 零法向梯度
            end
        end 
    end
    # 计算内部NS方程约束 - 硬约束
    Threads.@threads for i in 2:nx-1
        Threads.@threads for j in 2:ny-1
            # 评估点值
            U = up[i, j]
            V = vp[i, j]
            # 评估导数值
            Ux = fx(up[i-1, j], up[i+1, j], dx)
            Ux_up = fx_up(U, up[i-1, j], up[i+1, j], dx)
            Uy_up = fx_up(V, vp[i, j-1], vp[i, j+1], dy)
            Uxx = fxx(U, up[i-1, j], up[i+1, j], dx)
            Uyy = fxx(U, up[i, j-1], up[i, j+1], dy)
            Vx_up = fx_up(V, vp[i-1, j], vp[i+1, j], dx)
            Vy = fx(vp[i, j-1], vp[i, j+1], dy)
            Vy_up = fx_up(V, vp[i, j-1], vp[i, j+1], dy)
            Vxx = fxx(V, vp[i-1, j], vp[i+1, j], dx)
            Vyy = fxx(V, vp[i, j-1], vp[i, j+1], dy)
            Px = fx(pp[i-1, j], pp[i+1, j], dx)
            Py = fx(pp[i, j-1], pp[i, j+1], dy)

            Loss_mass[i, j] = Ux + Vy
            Loss_momentx[i, j] = rho*(U*Ux_up + V*Uy_up) + Px - mu*(Uxx + Uyy)
            Loss_momenty[i, j] = rho*(U*Vx_up + V*Vy_up) + Py - mu*(Vxx + Vyy)
        end
    end
    LossBond = @views vcat(Loss_bc_p_points,
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
                   Loss_bc_p_right)
    LossInter = @views vcat(
                    vec(Loss_momentx[2:nx-1, 2:ny-1]), 
                    vec(Loss_momenty[2:nx-1, 2:ny-1]),
                    vec(Loss_mass[2:nx-1, 2:ny-1]))
    # 实际返回向量
    Vecs = @views vcat(LossInter, LossBond)
    L2 = (sum(Vecs.^2)/length(Vecs))^0.5
    # 打印误差
    iterate[1] = iterate[1] + 1
    if iterate[1] % n_print == 0 || iterate[1] == 1
        @printf("iterate: %.1e \tL2 loss: %.4e \n", iterate[1], L2)
    end
    # 设置终止条件
    if iterate[1] < maxiters_initial
        return Vecs
    else
        return zeros(length(Vecs))
    end    
end

# 逐次逼近求解
for i in 1:N_steps+1
    global flat, iterate, model_u, model_v, model_p
    iterate = [0]
    Rein = paramsinputs[i][2]
    println()
    println("## Solving with nonlinear least squares, Re = $(Rein) ##")
    println("# number of DOFs is $(length(flat)) #")
    prob = NonlinearProblem(Fsolve, flat, paramsinputs[i])
    @time sol = solve(prob, LeastSquaresOptimJL(:lm, autodiff = :central))
    #@time sol = solve(prob, CMINPACK(method = :lmdif))
    flat = sol.u

    # 参数提取并重构解
    restored_vectors = split_vectors(flat, lengths)
    # 显式地提取三个向量
    out_params_u = restored_vectors[1]
    out_params_v = restored_vectors[2]
    out_params_p = restored_vectors[3]

    model_u = rebuild_u(out_params_u)
    model_v = rebuild_v(out_params_v)
    model_p = rebuild_p(out_params_p)

    plot_nx = 100
    plot_ny = 100
    plot_x = range(Lx, Rx, length=plot_nx)
    plot_y = range(Ly, Ry, length=plot_ny)
    u_num = ones(plot_nx, plot_ny)
    v_num = ones(plot_nx, plot_ny)
    velocity_num = ones(plot_nx, plot_ny)
    p_num = ones(plot_nx, plot_ny)
    for i in eachindex(plot_x)
        for j in eachindex(plot_y)
            u_num[i, j] = model_u([plot_x[i], plot_y[j]])[1]
            v_num[i, j] = model_v([plot_x[i], plot_y[j]])[1]
            velocity_num[i, j] = (u_num[i, j]^2 + v_num[i, j]^2)^0.5
            p_num[i, j] = model_p([plot_x[i], plot_y[j]])[1]
        end
    end
    u_num = u_num'
    v_num = v_num'
    velocity_num = velocity_num'
    p_num = p_num'
    CSV.write("u_num_steady.csv", DataFrame(u_num, :auto))
    CSV.write("v_num_steady.csv", DataFrame(v_num, :auto))
    CSV.write("velocity_num_steady.csv", DataFrame(velocity_num, :auto))
    CSV.write("p_num_steady.csv", DataFrame(p_num, :auto))
    p1 = plot(plot_x, plot_y, u_num, linetype =:contourf, title = "u (m/s)", xlabel = "x (m)", ylabel = "y (m)", color =:turbo)
    p2 = plot(plot_x, plot_y, v_num, linetype =:contourf, title = "v (m/s)", xlabel = "x (m)", ylabel = "y (m)", color =:turbo)
    p3 = plot(plot_x, plot_y, velocity_num, linetype =:contourf, title = "velocity (m/s)", xlabel = "x (m)", ylabel = "y (m)", color =:turbo)
    p4 = plot(plot_x, plot_y, p_num, linetype =:contourf, title = "p (Pa)", xlabel = "x (m)", ylabel = "y (m)", color =:turbo)
    # 使用layout参数组合这些图形
    combined_plot = plot(p1, p2, p3, p4, layout = (2, 2), size = (800, 600))
    # 显示或保存图形
    display(combined_plot)  # 显示图形
end