using Flux
using NonlinearSolve, LinearSolve, MINPACK
using Plots, Printf
using CSV, DataFrames

nx = 21
ny = 21
num_layers = 2 # output_dim
num_neurals = 13 # neurals_scale
## 参数设定
input_dims = 2 # x,y
output_dims = 1 # u,v,p
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
# 计算设定
# 设定雷诺数的变化范围

# 设定步数
N_steps = 0  # 将Re_min到Re_max分成N份（包括两端点）
n_print = 10000 # 每个时间步打印间隔数
maxiters_initial = 10000000 # 预处理最大训练步数
params = (rho, Re, v_slip, De, n_print)
iterate = [0] # 迭代符号
L2 = [1.0] # 残差记录符号

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
function fx_up(Uflow, f, fl, fr, delta) # 一阶迎风差分
    if Uflow > 0.0
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
    Loss_bc_p_points[1] = pp[nx, ny] - 0.0

    # 边界条件约束
    Threads.@threads for i in eachindex(x)
        @inbounds begin
            # 计算下边界条件 - 硬约束
            Loss_bc_u_down[i] = up[i, 1] - 0.0
            Loss_bc_v_down[i] = vp[i, 1] - 0.0
            Loss_bc_p_down[i] = model_p([x[i],y[1]-dy])[1] - pp[i, 2] # 零法向梯度
            # 计算上边界条件 - 硬约束
            Loss_bc_u_up[i] = up[i, ny] - v_slip
            Loss_bc_v_up[i] = vp[i, ny] - 0.0
            Loss_bc_p_up[i] = model_p([x[i],y[ny]+dy])[1] - pp[i, ny-1] # 零法向梯度
        end
    end
    Threads.@threads for j in eachindex(y)
        if j == 1 || j == ny # 上下底边防止过约束
            nothing
        else
            @inbounds begin
                # 计算左边界条件 - 硬约束
                Loss_bc_u_left[j] = up[1, j] - 0.0
                Loss_bc_v_left[j] = vp[1, j] - 0.0
                Loss_bc_p_left[j] = model_p([x[1]-dx,y[j]])[1] - pp[2, j] # 零法向梯度
                # 计算右边界条件 - 硬约束
                Loss_bc_u_right[j] = up[nx, j] - 0.0
                Loss_bc_v_right[j] = vp[nx, j] - 0.0
                Loss_bc_p_right[j] = model_p([x[nx]+dx,y[j]])[1] - pp[nx-1, j] # 零法向梯度
            end
        end 
    end
    
    # 计算内部NS方程约束 - 硬约束
    Threads.@threads for i in 1:nx
        Threads.@threads for j in 1:ny# 评估点值
            U = up[i, j]
            V = vp[i, j]
            P = pp[i, j]

            UL = 0.0
            VL = 0.0
            PL = 0.0
            UD = 0.0
            VD = 0.0
            PD = 0.0
            UR = 0.0
            VR = 0.0
            PR = 0.0
            UW = 0.0
            VW = 0.0
            PW = 0.0

            if j == 1 # 底边
                if i == 1 # 左点
                    UL = model_u([x[i]-dx,y[j]])[1]
                    VL = model_v([x[i]-dx,y[j]])[1]
                    PL = model_p([x[i]-dx,y[j]])[1]
    
                    UD = model_u([x[i],y[j]-dy])[1]
                    VD = model_v([x[i],y[j]-dy])[1]
                    PD = model_p([x[i],y[j]-dy])[1]
                    
                    UW = up[i, j+1]
                    VW = vp[i, j+1]
                    PW = pp[i, j+1]
                    
                    UR = up[i+1, j]
                    VR = vp[i+1, j]
                    PR = pp[i+1, j]
                    
                elseif i == nx # 右点
                    UR = model_u([x[i]+dx,y[j]])[1]
                    VR = model_v([x[i]+dx,y[j]])[1]
                    PR = model_p([x[i]+dx,y[j]])[1]

                    UD = model_u([x[i],y[j]-dy])[1]
                    VD = model_v([x[i],y[j]-dy])[1]
                    PD = model_p([x[i],y[j]-dy])[1]
                    
                    UL = up[i-1, j]
                    VL = vp[i-1, j]
                    PL = pp[i-1, j]

                    UW = up[i, j+1]
                    VW = vp[i, j+1]
                    PW = pp[i, j+1]
                else # 底边中间
                    UD = model_u([x[i],y[j]-dy])[1]
                    VD = model_v([x[i],y[j]-dy])[1]
                    PD = model_p([x[i],y[j]-dy])[1]

                    UR = up[i+1, j]
                    VR = vp[i+1, j]
                    PR = pp[i+1, j]

                    UW = up[i, j+1]
                    VW = vp[i, j+1]
                    PW = pp[i, j+1]
                    
                    UL = up[i-1, j]
                    VL = vp[i-1, j]
                    PL = pp[i-1, j]
                end
            elseif j == ny # 上边
                if i == 1 # 左点
                    UL = model_u([x[i]-dx,y[j]])[1]
                    VL = model_v([x[i]-dx,y[j]])[1]
                    PL = model_p([x[i]-dx,y[j]])[1]
                    
                    UW = model_u([x[i],y[j]+dy])[1]
                    VW = model_v([x[i],y[j]+dy])[1]
                    PW = model_p([x[i],y[j]+dy])[1]
                    
                    UR = up[i+1, j]
                    VR = vp[i+1, j]
                    PR = pp[i+1, j]
                    
                    UD = up[i, j-1]
                    VD = vp[i, j-1]
                    PD = pp[i, j-1]
                elseif i == nx # 右点
                    UR = model_u([x[i]+dx,y[j]])[1]
                    VR = model_v([x[i]+dx,y[j]])[1]
                    PR = model_p([x[i]+dx,y[j]])[1]

                    UW = model_u([x[i],y[j]+dy])[1]
                    VW = model_v([x[i],y[j]+dy])[1]
                    PW = model_p([x[i],y[j]+dy])[1]

                    UL = up[i-1, j]
                    VL = vp[i-1, j]
                    PL = pp[i-1, j]
                    
                    UD = up[i, j-1]
                    VD = vp[i, j-1]
                    PD = pp[i, j-1]
                else # 上边中间
                    UW = model_u([x[i],y[j]+dy])[1]
                    VW = model_v([x[i],y[j]+dy])[1]
                    PW = model_p([x[i],y[j]+dy])[1]

                    UL = up[i-1, j]
                    VL = vp[i-1, j]
                    PL = pp[i-1, j]
                    
                    UD = up[i, j-1]
                    VD = vp[i, j-1]
                    PD = pp[i, j-1]
                    
                    UR = up[i+1, j]
                    VR = vp[i+1, j]
                    PR = pp[i+1, j]
                end
            else # 中间
                if i == 1 # 左边
                    UL = model_u([x[i]-dx,y[j]])[1]
                    VL = model_v([x[i]-dx,y[j]])[1]
                    PL = model_p([x[i]-dx,y[j]])[1]
                    
                    UW = up[i, j+1]
                    VW = vp[i, j+1]
                    PW = pp[i, j+1]
                    
                    UR = up[i+1, j]
                    VR = vp[i+1, j]
                    PR = pp[i+1, j]
                        
                    UD = up[i, j-1]
                    VD = vp[i, j-1]
                    PD = pp[i, j-1]
                elseif i == nx # 右边
                    UR = model_u([x[i]+dx,y[j]])[1]
                    VR = model_v([x[i]+dx,y[j]])[1]
                    PR = model_p([x[i]+dx,y[j]])[1]

                    UW = up[i, j+1]
                    VW = vp[i, j+1]
                    PW = pp[i, j+1]

                    UL = up[i-1, j]
                    VL = vp[i-1, j]
                    PL = pp[i-1, j]
                    
                    UD = up[i, j-1]
                    VD = vp[i, j-1]
                    PD = pp[i, j-1]
                else # 内部
                    UL = up[i-1, j]
                    VL = vp[i-1, j]
                    PL = pp[i-1, j]
                    UD = up[i, j-1]
                    VD = vp[i, j-1]
                    PD = pp[i, j-1]
                    UR = up[i+1, j]
                    VR = vp[i+1, j]
                    PR = pp[i+1, j]
                    UW = up[i, j+1]
                    VW = vp[i, j+1]
                    PW = pp[i, j+1]
                end
            end
            
            # 评估导数值
            Ux = fx(UL, UR, dx) 
            Vy = fx(VD, VW, dy)
            
            Ux_up = fx_up(U, U, UL, UR, dx)
            Uxx = fxx(U, UL, UR, dx)
            Uy_up = fx_up(V, U, UD, UW, dy)
            Uyy = fxx(U, UD, UW, dy) 
    
            Vx_up = fx_up(U, V, VL, VR, dx)
            Vxx = fxx(V, VL, VR, dx)
            Vy_up = fx_up(U, V, VD, VW, dy)
            Vyy = fxx(V, VD, VW, dy)
            
            Px = fx(PL, PR, dx)
            Py = fx(PD, PW, dy)
                        
            # 组装方程
            Loss_momentx[i, j] = (U*Ux_up + V*Uy_up) + Px/rho - mu/rho*(Uxx + Uyy) # U动量方程
            Loss_momenty[i, j] = (U*Vx_up + V*Vy_up) + Py/rho - mu/rho*(Vxx + Vyy) # V动量方程
            Loss_mass[i, j] = rho*(Ux + Vy) # 连续性方程
            #OutputU[i, j, 3] = (Pxx + Pyy)/rho + (Ux^2 + Vy^2 + 2*Uy*Vx) # 压力泊松方程
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
                    vec(Loss_momentx), 
                    vec(Loss_momenty),
                    vec(Loss_mass))
    # 实际返回向量
    Vecs = @views vcat(LossInter, LossBond)
    L2[1] = (sum(Vecs.^2)/length(Vecs))^0.5
    
    # 打印误差
    iterate[1] = iterate[1] + 1
    if iterate[1] % n_print == 0 || iterate[1] == 1
        @printf("iterate: %.5e \tL2 loss: %.4e \n", iterate[1], L2[1])
    elseif L2[1] <= 1e-5
        @printf("iterate: %.5e \tL2 loss: %.4e \n", iterate[1], L2[1])
    else
        nothing
    end
    # 设置终止条件
    if iterate[1] < maxiters_initial
        return Vecs
    else
        return zeros(length(Vecs))
    end    
end

println()
println("## Solving with nonlinear least squares, Re = $(Re) ##")
println("# number of DOFs is $(length(flat)) #")
println("# number of CONs is $(length(Fsolve(flat, params))) #")
prob = NonlinearProblem(Fsolve, flat, params)
#@time sol = solve(prob, LeastSquaresOptimJL(:lm, autodiff = :central))
@time sol = solve(prob, CMINPACK(method = :lmdif))
#@time sol = solve(prob, NonlinearSolve.GaussNewton(; autodiff = AutoSparse(AutoFiniteDiff())))

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

err_u = ones(plot_nx, plot_ny)
err_v = ones(plot_nx, plot_ny)
err_velocity = ones(plot_nx, plot_ny)
err_p = ones(plot_nx, plot_ny)

file_path_u = "Newton_u_num_steady.csv"
file_path_v = "Newton_v_num_steady.csv"
file_path_velocity = "Newton_velocity_num_steady.csv"
file_path_p = "Newton_p_num_steady.csv"
df_u = CSV.read(file_path_u, DataFrame)
df_v = CSV.read(file_path_v, DataFrame)
df_velocity = CSV.read(file_path_velocity, DataFrame)
df_p = CSV.read(file_path_p, DataFrame)
u_ANA = Matrix(df_u)
v_ANA = Matrix(df_v)
velocity_ANA = Matrix(df_velocity)
p_ANA = Matrix(df_p)

for i in eachindex(plot_x)
    for j in eachindex(plot_y)
        u_num[i, j] = model_u([plot_x[i], plot_y[j]])[1]
        v_num[i, j] = model_v([plot_x[i], plot_y[j]])[1]
        velocity_num[i, j] = (u_num[i, j]^2 + v_num[i, j]^2)^0.5
        p_num[i, j] = model_p([plot_x[i], plot_y[j]])[1]
    
        err_u[i, j] = u_num[i, j] - u_ANA[i, j]
        err_v[i, j] = v_num[i, j] - v_ANA[i, j]
        err_velocity[i, j] = velocity_num[i, j] - velocity_ANA[i, j]
        err_p[i,j] = p_num[i, j] - p_ANA[i, j]
    end
end
u_num = u_num'
v_num = v_num'
velocity_num = velocity_num'
p_num = p_num'
u_ANA = u_ANA
v_ANA = v_ANA
velocity_ANA = velocity_ANA
p_ANA = p_ANA
CSV.write("LDC_u_num_steady.csv", DataFrame(u_num, :auto))
CSV.write("LDC_v_num_steady.csv", DataFrame(v_num, :auto))
CSV.write("LDC_velocity_num_steady.csv", DataFrame(velocity_num, :auto))
CSV.write("LDC_p_num_steady.csv", DataFrame(p_num, :auto))
p1 = contourf(plot_x, plot_y, u_num, lw = 0, levels=20, title = "u (m/s)", color =:turbo)
p2 = contourf(plot_x, plot_y, v_num, lw = 0, levels=20, title = "v (m/s)", color =:turbo)
p3 = contourf(plot_x, plot_y, velocity_num, lw = 0, levels=20, title = "velocity (m/s)", color =:turbo)
p4 = contourf(plot_x, plot_y, p_num, lw = 0, levels=20, title = "p (Pa)", color =:turbo)

p5 = contourf(plot_x, plot_y, u_ANA, lw = 0, levels=20, color =:turbo)
p6 = contourf(plot_x, plot_y, v_ANA, lw = 0, levels=20, color =:turbo)
p7 = contourf(plot_x, plot_y, velocity_ANA, lw = 0, levels=20, color =:turbo)
p8 = contourf(plot_x, plot_y, p_ANA, lw = 0, levels=20, color =:turbo)

p9 = contourf(plot_x, plot_y, err_u, lw = 0, levels=20, color =:turbo)
p10 = contourf(plot_x, plot_y, err_v, lw = 0, levels=20, color =:turbo)
p11 = contourf(plot_x, plot_y, err_velocity, lw = 0, color =:turbo)
p12 = contourf(plot_x, plot_y, err_p, lw = 0, levels=20, color =:turbo)
# 使用layout参数组合这些图形
combined_plot = plot(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12,  layout = (3, 4), size = (1200, 500))
# 显示或保存图形
display(combined_plot)  # 显示图形
L2_matrix_u = (sum(err_u.^2)/(nx*ny))^0.5
L2_matrix_v = (sum(err_v.^2)/(nx*ny))^0.5
L2_matrix_velocity = (sum(err_velocity.^2)/(nx*ny))^0.5
L2_matrix_p = (sum(err_p.^2)/(nx*ny))^0.5
println(L2_matrix_u)
println(L2_matrix_v)
println(L2_matrix_velocity)
println(L2_matrix_p)


