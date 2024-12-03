using Flux
using NonlinearSolve, LeastSquaresOptim
using Plots, Printf
using CSV, DataFrames

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

## 创建模型实例
input_dims = 2 # x,y
output_dims = 3 # u,v,p
num_layers = 3 # output_dim
num_neurals = 8 # mesh_scale
layers_neurals = fill(num_neurals, num_layers + 1)
model = f64(create_network(input_dims, output_dims, layers_neurals))
println(model)

## 参数设定
# 计算控制常数
rho = 1.0 # 密度
Re = 1000.0 # 雷诺数
v_slip = 1.0 # 滑移速度
De = 1.0 # 水力直径
train_print = 500 # 训练步打印间隔数
n_print = 10000 # 每个时间步打印间隔数
dt = 0.5
paramsinput = (rho, Re, v_slip, De, n_print, dt)
# 其他参数
t_start = 0.0 # 初始时间
t_end = 20.0 # 终止时间
epochs = 2000 # 初始逼近步数
plot_nx = 100 # x绘图间隔
plot_ny = 100 # y绘图间隔
# 网格设定
Lx = 0.0
Rx = 1.0
nx = 10
dx = (Rx - Lx)/(nx-1)
x = range(Lx, Rx, length=nx)
Ly = 0.0
Ry = 1.0
ny = 10
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

# 定义初始条件加载函数
function smoothStep(x; start=t_start, ends=10, steepness=10)
    # 使函数在x接近start时值接近0，在x接近end时值接近1
    normalized_x = (x - start) / (ends - start)
    return 1 / (1 + exp(-steepness * (normalized_x - 0.5)))
end

##  逼近初始值
# 初始条件
u_initial = zeros(nx, ny)
v_initial = zeros(nx, ny)
p_initial = zeros(nx, ny)
println("== Initializing for the initial values ==")
# 定义损失函数，这里的目标是使网络输出接近初始场
function lossfunc()
    loss_value = 0.0
    for i in eachindex(x)
        for j in eachindex(y)
            prediction = model([x[i], y[j]])
            target = [u_initial[i, j], v_initial[i, j], p_initial[i, j]]
            loss_value += sum(abs.(prediction - target))
        end
    end
    return loss_value
end
opt = Flux.Adam()  # 使用Adam优化器
for i in 1:epochs
    current_loss = lossfunc()  # 计算当前损失
    if i%train_print == 0
        println("Epoch $i, Loss: $current_loss")
    end
    # 进行一次优化迭代
    Flux.train!(() -> lossfunc(), Flux.params(model), Iterators.repeated((), 1), opt)
end

## 非线性最小二乘求解
println()
println("== Solving with nonlinear least squares ==")

# 提取模型参数
flat, rebuild = Flux.destructure(model)
u_sol = zeros(nx, ny)
v_sol = zeros(nx, ny)
p_sol = zeros(nx, ny)

times = t_start
function Transient_calculating(times,model)
    while times < t_end
        @printf("Times: %.1f \tTime_end: %.1f\n", times, t_end)
        # 获取上一个时刻解
        @sync begin
            for i in eachindex(x)
                @async begin
                    Threads.@threads for j in eachindex(y)
                        u_sol[i, j],v_sol[i, j], p_sol[i,j] = model([x[i], y[j]])
                    end
                end
            end
        end
        # 迭代记号
        iterate = [0]
        # 定义稳态Jacobi-Levenberg-Marquardt迭代函数
        function Fsolve(init_params,params_in)
            rho = params_in[1]
            Re = params_in[2]
            v_slip = params_in[3]
            De = params_in[4]
            n_print = params_in[5]
            dt = params_in[6]
            mu = rho*v_slip*De/Re
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
            # 计算下边界条件 - 硬约束
            Threads.@threads for i in eachindex(x)
                Loss_bc_u_down[i] = u(x[i], y[1]) - 0.0
                Loss_bc_v_down[i] = v(x[i], y[1]) - 0.0
                Loss_bc_p_down[i] = p_y(x[i], y[1]) - 0.0
            end
            # 计算上边界条件 - 硬约束
            Threads.@threads for i in eachindex(x)
                Loss_bc_u_up[i] = u(x[i], y[ny]) - v_slip*smoothStep(times)
                Loss_bc_v_up[i] = v(x[i], y[ny]) - 0.0
                Loss_bc_p_up[i] = p_y(x[i], y[ny]) - 0.0
            end
            # 计算左边界条件 - 硬约束
            Threads.@threads for i in eachindex(y)
                if i == ny
                    nothing # 不要过约束
                else
                    Loss_bc_u_left[i] = u(x[1], y[i]) - 0.0
                    Loss_bc_v_left[i] = v(x[1], y[i]) - 0.0
                    Loss_bc_p_left[i] = p_x(x[1], y[i]) - 0.0
                end 
            end
            # 计算右边界条件 - 硬约束
            Threads.@threads for i in eachindex(y)
                if i == ny
                    nothing # 不要过约束
                else
                    Loss_bc_u_right[i] = u(x[nx], y[i]) - 0.0
                    Loss_bc_v_right[i] = v(x[nx], y[i]) - 0.0
                    Loss_bc_p_right[i] = p_x(x[nx], y[i]) - 0.0
                end
            end
            # 计算内部NS方程约束 - 硬约束
            @sync begin
                for i in 2:nx-1
                    @async begin
                        Threads.@threads for j in 2:ny-1
                            U_old = u_sol[i, j]
                            V_old = v_sol[i, j]
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
                            Loss_momentx[i, j] = rho*(((U*Ux_upwind + V*Uy_upwind) + Px - mu*(Uxx + Uyy))*dt + U - U_old)
                            Loss_momenty[i, j] = rho*(((U*Vx_upwind + V*Vy_upwind) + Py - mu*(Vxx + Vyy))*dt + V - V_old)
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
            LossInter = cat(vec(Loss_mass[2:nx-1, 2:ny-1]),
                            vec(Loss_momentx[2:nx-1, 2:ny-1]), 
                            vec(Loss_momenty[2:nx-1, 2:ny-1]), 
                            dims = 1)
            # 实际返回向量
            Vecs = cat(LossBond, LossInter, dims = 1)
            L2 = (sum(Vecs.^2)/length(Vecs))^0.5
            # 打印误差
            iterate[1] = iterate[1] + 1
            if iterate[1] % n_print == 0 || iterate[1] == 1
                @printf("iterate: %d \tL2 loss: %.4e \n", iterate[1], L2)
            end
            return Vecs
        end
        # 求解
        prob = NonlinearProblem(Fsolve, flat, paramsinput)
        #@time sol = solve(prob, CMINPACK(method = :lmdif))
        @time sol = solve(prob, LeastSquaresOptimJL())
        # 重构模型
        model = f64(rebuild(sol.u))
        # 时间步进
        times = times + dt
        # 绘图初始化
        plot_x = range(Lx, Rx, length=plot_nx)
        plot_y = range(Ly, Ry, length=plot_ny)
        u_num = ones(plot_nx, plot_ny)
        v_num = ones(plot_nx, plot_ny)
        velocity_num = ones(plot_nx, plot_ny)
        p_num = ones(plot_nx, plot_ny)
        @sync begin
            for i in eachindex(plot_x)
                @async begin
                    Threads.@threads for j in eachindex(plot_y)
                        u_num[i, j] = model([plot_x[i], plot_y[j]])[1]
                        v_num[i, j] = model([plot_x[i], plot_y[j]])[2]
                        velocity_num[i, j] = (u_num[i, j]^2 + v_num[i, j]^2)^0.5
                        p_num[i, j] = model([plot_x[i], plot_y[j]])[3]
                    end
                end
            end
        end
        u_num = u_num'
        v_num = v_num'
        velocity_num = velocity_num'
        p_num = p_num'
        CSV.write("u_num_$times.csv", DataFrame(u_num, :auto))
        CSV.write("v_num_$times.csv", DataFrame(v_num, :auto))
        CSV.write("velocity_num_$times.csv", DataFrame(velocity_num, :auto))
        CSV.write("p_num_$times.csv", DataFrame(p_num, :auto))
        p1 = plot(plot_x, plot_y, u_num, linetype =:contourf, 
                    title = "u (m/s), time = $times", xlabel = "x (m)", ylabel = "y (m)")
        p2 = plot(plot_x, plot_y, v_num, linetype =:contourf, 
                    title = "v (m/s), time = $times", xlabel = "x (m)", ylabel = "y (m)")
        p3 = plot(plot_x, plot_y, velocity_num, linetype =:contourf, 
                    title = "velocity (m/s), time = $times", xlabel = "x (m)", ylabel = "y (m)")
        p4 = plot(plot_x, plot_y, p_num, linetype =:contourf, 
                    title = "p (Pa), time = $times", xlabel = "x (m)", ylabel = "y (m)")
        # 使用layout参数组合这些图形
        combined_plot = plot(p1, p2, p3, p4, layout = (2, 2), size = (800, 600))
        # 显示或保存图形
        display(combined_plot)  # 显示图形
    end
    return model, times
end 
model, times = Transient_calculating(times,model)
