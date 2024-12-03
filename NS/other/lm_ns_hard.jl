using Flux
using NonlinearSolve, MINPACK, LeastSquaresOptim, LinearSolve
using Plots, Printf
using CSV, DataFrames

# 定义差分算子
function fx_up(U, f, fl, fr, delta) # 一阶迎风差分
    if U > 0.0
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

nx = 30
ny = 30
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
n_print = 5000 # 每个时间步打印间隔数
maxiters_initial = 1000000 # 预处理最大训练步数
params = (rho, Re, v_slip, De, n_print)
iterate = [0] # 迭代符号

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
Loss_mass = zeros(nx-1, ny-1)
Loss_momentx = zeros(nx-1, ny-1)
Loss_momenty = zeros(nx-1, ny-1)
# 数据解点空间
up = zeros(nx, ny)
vp = zeros(nx, ny)
pp = zeros(nx, ny)
# 向量拼接传入
combined_U = [vec(up); vec(vp); vec(pp)]
# 获取矩阵的行数和列数
rows, cols = size(up) # 假设所有矩阵大小相同

# 定义稳态Jacobi-Levenberg-Marquardt迭代函数
function Fsolve(combined_U,params_in)
    rho = params_in[1]
    Re = params_in[2]
    v_slip = params_in[3]
    De = params_in[4]
    n_print = params_in[5]
    mu = rho*v_slip*De/Re
    # 从向量中恢复矩阵
    vecu_recovered = @view combined_U[1:rows*cols]
    vecv_recovered = @view combined_U[(rows*cols+1):(2*rows*cols)]
    vecp_recovered = @view combined_U[(2*rows*cols+1):end]
    
    # 重塑向量回到矩阵
    up = reshape(vecu_recovered, (rows, cols))
    vp = reshape(vecv_recovered, (rows, cols))
    pp = reshape(vecp_recovered, (rows, cols))

    # 计算压力点约束
    Loss_bc_p_points[1] = pp[nx, ny] - 0.0

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
            Ux_up = fx_up(U, up[i, j], up[i-1, j], up[i+1, j], dx)
            Uy_up = fx_up(V, up[i, j], up[i, j-1], up[i, j+1], dy)
            Uxx = fxx(U, up[i-1, j], up[i+1, j], dx)
            Uyy = fxx(U, up[i, j-1], up[i, j+1], dy)
            Vx_up = fx_up(U, vp[i, j], vp[i-1, j], vp[i+1, j], dx)
            Vy = fx(vp[i, j-1], vp[i, j+1], dy)
            Vy_up = fx_up(V, vp[i, j], vp[i, j-1], vp[i, j+1], dy)
            Vxx = fxx(V, vp[i-1, j], vp[i+1, j], dx)
            Vyy = fxx(V, vp[i, j-1], vp[i, j+1], dy)
            Px = fx(pp[i-1, j], pp[i+1, j], dx)
            Py = fx(pp[i, j-1], pp[i, j+1], dy)

            Loss_mass[i-1, j] = Ux + Vy
            Loss_momentx[i-1, j] = rho*(U*Ux_up + V*Uy_up) + Px - mu*(Uxx + Uyy)
            Loss_momenty[i-1, j] = rho*(U*Vx_up + V*Vy_up) + Py - mu*(Vxx + Vyy)
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

println()
println("## Solving with nonlinear least squares, Re = $(Re) ##")
println("# number of DOFs is $(length(combined_U)) #")
println("# number of CONs is $(length(Fsolve(combined_U, params))) #")
prob = NonlinearProblem(Fsolve, combined_U, params)

#@time sol = solve(prob, LeastSquaresOptimJL(:lm, autodiff = :central))
#@time sol = solve(prob, NonlinearSolve.LevenbergMarquardt(; linsolve = QRFactorization(), autodiff = AutoSparse(AutoFiniteDiff())))
@time sol = solve(prob, CMINPACK(method = :lmdif))
flat = sol.u

u_matrix = @view flat[1:rows*cols]
v_matrix = @view flat[(rows*cols+1):(2*rows*cols)]
p_matrix = @view flat[(2*rows*cols+1):end]

# 重塑向量回到矩阵
u_num = reshape(u_matrix, (rows, cols))
v_num = reshape(v_matrix, (rows, cols))
p_num = reshape(p_matrix, (rows, cols))
velocity_num  = (u_num.^2 + v_num.^2).^0.5

u_num = u_num'
v_num = v_num'
velocity_num = velocity_num'
p_num = p_num'

plot_x = range(Lx, Rx, length=nx)
plot_y = range(Ly, Ry, length=ny)
p1 = plot(plot_x, plot_y, u_num, linetype =:contourf, title = "u (m/s)", xlabel = "x (m)", ylabel = "y (m)", color =:turbo)
p2 = plot(plot_x, plot_y, v_num, linetype =:contourf, title = "v (m/s)", xlabel = "x (m)", ylabel = "y (m)", color =:turbo)
p3 = plot(plot_x, plot_y, velocity_num, linetype =:contourf, title = "velocity (m/s)", xlabel = "x (m)", ylabel = "y (m)", color =:turbo)
p4 = plot(plot_x, plot_y, p_num, linetype =:contourf, title = "p (Pa)", xlabel = "x (m)", ylabel = "y (m)", color =:turbo)
combined_plot = plot(p1, p2, p3, p4, layout = (2, 2), size = (800, 600))
display(combined_plot)  # 显示图形

CSV.write("LM_u_num_steady.csv", DataFrame(u_num, :auto))
CSV.write("LM_v_num_steady_LM.csv", DataFrame(v_num, :auto))
CSV.write("LM_velocity_num_LM_steady.csv", DataFrame(velocity_num, :auto))
CSV.write("LM_p_num_steady_LM.csv", DataFrame(p_num, :auto))
#=


# 使用layout参数组合这些图形
# 显示或保存图形
=#