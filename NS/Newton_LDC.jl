using NonlinearSolve, LinearSolve
using Plots, Printf
using CSV, DataFrames

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

nx = 100
ny = 100

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
maxiters_initial = 20 # 预处理最大训练步数
params = (rho, Re, v_slip, De, n_print)
iterate = [0] # 迭代符号
L2 = [1.0] # 残差记录符号

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

# 数据解点空间
up = zeros(nx, ny)
vp = zeros(nx, ny)
pp = zeros(nx, ny)

# 向量拼接传入
combined_U = cat(up, vp, pp; dims=3)
# 获取矩阵的行数和列数
rows, cols = size(up) # 假设所有矩阵大小相同

# 定义稳态Jacobi-Levenberg-Marquardt迭代函数
function Fsolve(OutputU, combined_U, params_in)
    rho = params_in[1]
    Re = params_in[2]
    v_slip = params_in[3]
    De = params_in[4]
    n_print = params_in[5]
    mu = rho*v_slip*De/Re
    
    # 重塑矩阵
    up = combined_U[:,:,1]
    vp = combined_U[:,:,2]
    pp = combined_U[:,:,3]
    
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
                    UL = 2*Loss_bc_u_left[j] - up[i+1, j]
                    VL = 2*Loss_bc_v_left[j] - vp[i+1, j]
                    PL = pp[i, j]
    
                    UD = 2*Loss_bc_u_down[i] - up[i, j+1]
                    VD = 2*Loss_bc_v_down[i] - vp[i, j+1]
                    PD = pp[i, j]
                    
                    UW = up[i, j+1]
                    VW = vp[i, j+1]
                    PW = pp[i, j+1]
                    
                    UR = up[i+1, j]
                    VR = vp[i+1, j]
                    PR = pp[i+1, j]
                    
                elseif i == nx # 右点
                    UR = 2*Loss_bc_u_right[j] - up[i-1, j]
                    VR = 2*Loss_bc_v_right[j] - vp[i-1, j]
                    PR = pp[i, j]

                    UD = 2*Loss_bc_u_down[i] - up[i, j+1]
                    VD = 2*Loss_bc_v_down[i] - vp[i, j+1]
                    PD = pp[i, j]
                    
                    UL = up[i-1, j]
                    VL = vp[i-1, j]
                    PL = pp[i-1, j]

                    UW = up[i, j+1]
                    VW = vp[i, j+1]
                    PW = pp[i, j+1]
                else # 底边中间
                    UD = 2*Loss_bc_u_down[i] - up[i, j+1]
                    VD = 2*Loss_bc_v_down[i] - vp[i, j+1]
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
                    UL = 2*Loss_bc_u_left[j] - up[i+1, j]
                    VL = 2*Loss_bc_v_left[j] - vp[i+1, j]
                    PL = pp[i, j]
                    
                    UW = 2*v_slip - up[i, j-1] # 上无滑移边界条件
                    VW = 2*Loss_bc_v_up[i] - vp[i, j-1]
                    PW = pp[i, j]
                    
                    UR = up[i+1, j]
                    VR = vp[i+1, j]
                    PR = pp[i+1, j]
                    
                    UD = up[i, j-1]
                    VD = vp[i, j-1]
                    PD = pp[i, j-1]
                elseif i == nx # 右点
                    UR = 2*Loss_bc_u_right[j] - up[i-1, j]
                    VR = 2*Loss_bc_v_right[j] - vp[i-1, j]
                    PR = pp[i, j]

                    UW = 2*v_slip - up[i, j-1] # 上无滑移边界条件
                    VW = 2*Loss_bc_v_up[i] - vp[i, j-1]
                    PW = 2*0.0 - pp[i, j-1] # 压力点约束

                    UL = up[i-1, j]
                    VL = vp[i-1, j]
                    PL = pp[i-1, j]
                    
                    UD = up[i, j-1]
                    VD = vp[i, j-1]
                    PD = pp[i, j-1]
                else # 上边中间
                    UW = 2*v_slip - up[i, j-1] # 上无滑移边界条件
                    VW = 2*Loss_bc_v_up[i] - vp[i, j-1]
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
                    UL = 2*Loss_bc_u_left[j] - up[i+1, j]
                    VL = 2*Loss_bc_v_left[j] - vp[i+1, j]
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
                    UR = 2*Loss_bc_u_right[j] - up[i-1, j]
                    VR = 2*Loss_bc_v_right[j] - vp[i-1, j]
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

            #=
            # 局部定向差分算子保持稳定性
            if i == nx && j == ny
                #Px = fx_up(-1, P, PL, PR, dx)
                Py = fx_up(-1, P, PD, PW, dy)
            end
            =#
                        
            # 组装方程
            OutputU[i, j, 1] = (U*Ux_up + V*Uy_up) + Px/rho - mu/rho*(Uxx + Uyy) # U动量方程
            OutputU[i, j, 2] = (U*Vx_up + V*Vy_up) + Py/rho - mu/rho*(Vxx + Vyy) # V动量方程
            OutputU[i, j, 3] = rho*(Ux + Vy) # 连续性方程
            #OutputU[i, j, 3] = (Pxx + Pyy)/rho + (Ux^2 + Vy^2 + 2*Uy*Vx) # 压力泊松方程
        end
    end
    
    L2[1] = (sum(OutputU.^2)/length(OutputU))^0.5
    # 打印误差
    iterate[1] = iterate[1] + 1
    if iterate[1] % n_print == 0 || iterate[1] == 1
        @printf("iterate: %.5e \tL2 loss: %.4e \n", iterate[1], L2[1])
    elseif L2[1] <= 1e-5
        @printf("iterate: %.5e \tL2 loss: %.4e \n", iterate[1], L2[1])
    else
        nothing
    end   
end

println()
println("## Solving with nonlinear solvers, Re = $(Re) ##")
println("# number of DOFs is $(length(combined_U)), $(size(combined_U)) #")
prob = NonlinearProblem(Fsolve, combined_U, params)

# 性能类似
@time sol = solve(prob, NewtonRaphson(linsolve = KLUFactorization(); autodiff = AutoSparse(AutoFiniteDiff())))
#@time sol = solve(prob, RobustMultiNewton(; autodiff = AutoSparse(AutoFiniteDiff())))
#@time sol = solve(prob, TrustRegion(linsolve = KLUFactorization(); autodiff = AutoSparse(AutoFiniteDiff())))
#@time sol = solve(prob, NewtonRaphson(; autodiff = AutoSparse(AutoFiniteDiff())))
# 伪时间步进，精度高
#@time sol = solve(prob, PseudoTransient(alpha_initial = 0.1; autodiff = AutoSparse(AutoFiniteDiff())))
#@time sol = solve(prob, PseudoTransient())
# 瞬态法，很慢
#@time sol = solve(prob, DynamicSS(ROS3P()))
# 非线性最小二乘法
#@time sol = solve(prob, NonlinearSolve.LevenbergMarquardt(; autodiff = AutoSparse(AutoFiniteDiff())))
#@time sol = solve(prob, NonlinearSolve.GaussNewton(; autodiff = AutoSparse(AutoFiniteDiff())))

#=
## 代数多重网格预条件器
using AlgebraicMultigrid
function algebraicmultigrid(W, du, u, p, t, newW, Plprev, Prprev, solverdata)
    if newW === nothing || newW
        Pl = aspreconditioner(ruge_stuben(convert(AbstractMatrix, W)))
    else
        Pl = Plprev
    end
    Pl, nothing
end
@time sol = solve(prob, TrustRegion(linsolve = KrylovJL_GMRES(), precs = algebraicmultigrid, concrete_jac = true; autodiff = AutoSparse(AutoFiniteDiff())), maxiters = maxiters_initial)
=#
#=
## 代数多重网格jacobi光滑求解器
function algebraicmultigrid2(W, du, u, p, t, newW, Plprev, Prprev, solverdata)
    if newW === nothing || newW
        A = convert(AbstractMatrix, W)
        Pl = AlgebraicMultigrid.aspreconditioner(AlgebraicMultigrid.ruge_stuben(
            A, presmoother = AlgebraicMultigrid.Jacobi(rand(size(A, 1))),
            postsmoother = AlgebraicMultigrid.Jacobi(rand(size(A, 1)))))
    else
        Pl = Plprev
    end
    Pl, nothing
end
@time sol = solve(prob, RobustMultiNewton(linsolve = KrylovJL_GMRES(), precs = algebraicmultigrid2, concrete_jac = true; autodiff = AutoSparse(AutoFiniteDiff())), maxiters = maxiters_initial)
=#

flat = sol.u

u_matrix = flat[:,:,1]
v_matrix = flat[:,:,2]
p_matrix = flat[:,:,3]

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
p1 = contourf(plot_x, plot_y, u_num, lw = 0, levels=20, title = "u (m/s)", xlabel = "x (m)", ylabel = "y (m)", color =:turbo)
p2 = contourf(plot_x, plot_y, v_num, lw = 0, levels=20, title = "v (m/s)", xlabel = "x (m)", ylabel = "y (m)", color =:turbo)
p3 = contourf(plot_x, plot_y, velocity_num, lw = 0, levels=20, title = "velocity (m/s)", xlabel = "x (m)", ylabel = "y (m)", color =:turbo)
p4 = contourf(plot_x, plot_y, p_num, lw = 0, levels=20, title = "p (Pa)", xlabel = "x (m)", ylabel = "y (m)", color =:turbo)
combined_plot = plot(p1, p2, p3, p4, layout = (2, 2), size = (800, 675))
display(combined_plot)  # 显示图形

CSV.write("Newton_u_num_steady.csv", DataFrame(u_num, :auto))
CSV.write("Newton_v_num_steady.csv", DataFrame(v_num, :auto))
CSV.write("Newton_velocity_num_steady.csv", DataFrame(velocity_num, :auto))
CSV.write("Newton_p_num_steady.csv", DataFrame(p_num, :auto))
