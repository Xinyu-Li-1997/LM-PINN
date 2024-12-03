using Flux
using NonlinearSolve, MINPACK, LeastSquaresOptim, LinearSolve
using Plots, Printf
using CSV, DataFrames

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

nx = 50
ny = 50
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
n_print = 1000 # 每个时间步打印间隔数
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
# 内部条件
Loss_mass = zeros(nx, ny)
Loss_momentx = zeros(nx, ny)
Loss_momenty = zeros(nx, ny)
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
                    UL = Loss_bc_u_left[j]
                    VL = Loss_bc_v_left[j]
                    PL = pp[i, j]
    
                    UD = Loss_bc_u_down[i]
                    VD = Loss_bc_v_down[i]
                    PD = pp[i, j]
                    
                    UW = up[i, j+1]
                    VW = vp[i, j+1]
                    PW = pp[i, j+1]
                    
                    UR = up[i+1, j]
                    VR = vp[i+1, j]
                    PR = pp[i+1, j]
                    
                elseif i == nx # 右点
                    UR = Loss_bc_u_right[j]
                    VR = Loss_bc_v_right[j]
                    PR = pp[i, j]

                    UD = Loss_bc_u_down[i]
                    VD = Loss_bc_v_down[i]
                    PD = pp[i, j]
                    
                    UL = up[i-1, j]
                    VL = vp[i-1, j]
                    PL = pp[i-1, j]

                    UW = up[i, j+1]
                    VW = vp[i, j+1]
                    PW = pp[i, j+1]
                else # 底边中间
                    UD = Loss_bc_u_down[i]
                    VD = Loss_bc_v_down[i]
                    PD = pp[i, j]

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
                    UL = Loss_bc_u_left[j]
                    VL = Loss_bc_v_left[j]
                    PL = pp[i, j]
                    
                    UW = v_slip # 上无滑移边界条件
                    VW = Loss_bc_v_up[i]
                    PW = pp[i, j]
                    
                    UR = up[i+1, j]
                    VR = vp[i+1, j]
                    PR = pp[i+1, j]
                    
                    UD = up[i, j-1]
                    VD = vp[i, j-1]
                    PD = pp[i, j-1]
                elseif i == nx # 右点
                    UR = Loss_bc_u_right[j]
                    VR = Loss_bc_v_right[j]
                    PR = pp[i, j]

                    UW = v_slip # 上无滑移边界条件
                    VW = Loss_bc_v_up[i]
                    PW = 0.0 # 压力点约束

                    UL = up[i-1, j]
                    VL = vp[i-1, j]
                    PL = pp[i-1, j]
                    
                    UD = up[i, j-1]
                    VD = vp[i, j-1]
                    PD = pp[i, j-1]
                else # 上边中间
                    UW = v_slip # 上无滑移边界条件
                    VW = Loss_bc_v_up[i]
                    PW = pp[i, j]

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
                    UL = Loss_bc_u_left[j]
                    VL = Loss_bc_v_left[j]
                    PL = pp[i, j]
                    
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
                    UR = Loss_bc_u_right[j]
                    VR = Loss_bc_v_right[j]
                    PR = pp[i, j]

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
            Ux_up = fx_up(U, U, UL, UR, dx)
            Uxx = fxx(U, UL, UR, dx)
            Uy_up = fx_up(V, U, UD, UW, dy)
            Uyy = fxx(U, UD, UW, dy)
            Vy = fx(VD, VW, dy)
            Vx_up = fx_up(U, V, VL, VR, dx)
            Vxx = fxx(V, VL, VR, dx)
            Vy_up = fx_up(U, V, VD, VW, dy)
            Vyy = fxx(V, VD, VW, dy)
            Px = fx(PL, PR, dx)
            Py = fx(PD, PW, dy)
            # 组装方程
            Loss_momentx[i, j] = rho*(U*Ux_up + V*Uy_up) + Px - mu*(Uxx + Uyy)
            Loss_momenty[i, j] = rho*(U*Vx_up + V*Vy_up) + Py - mu*(Vxx + Vyy)
            Loss_mass[i, j] = Ux + Vy
        end
    end
    # 实际返回向量
    Vecs = @views vcat(
                    vec(Loss_momentx),
                    vec(Loss_momenty),
                    vec(Loss_mass))
    
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
#@time sol = solve(prob, CMINPACK(method = :lmdif))
#prob = NonlinearProblem(Fsolve, sol.u, params)
@time sol = solve(prob, NewtonRaphson(linsolve = KrylovJL_GMRES(); autodiff = AutoSparse(AutoFiniteDiff())))

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
