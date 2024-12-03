using NonlinearSolve
using LinearSolve
using Sundials
using ModelingToolkit
using MethodOfLines
using DomainSets
using Plots
@parameters x y
@variables u(..) v(..) p(..)
Dx = Differential(x)
Dy = Differential(y)
Dxx = Differential(x)^2
Dyy = Differential(y)^2

dx = 0.05
dy = 0.05
order = 2

# 常数
rho = 1.0 # 密度
v_slip = 1.0 # 滑移速度
De = 1.0 # 水力直径
Re = 100.0 # 雷诺数
Lx = 0.0
Rx = 1.0
Ly = 0.0
Ry = 1.0
mu = rho*v_slip*De/Re

# 算子定义
U = u(x, y)
V = v(x, y)
Ux = Dx(u(x, y))
Vx = Dx(v(x, y))
Uy = Dy(u(x, y))
Vy = Dy(v(x, y))
Uxx = Dxx(u(x, y))
Uyy = Dyy(u(x, y))
Vxx = Dxx(v(x, y))
Vyy = Dyy(v(x, y))
Px = Dx(p(x, y))
Py = Dy(p(x, y))
Ix = Integral(x in DomainSets.ClosedInterval(Lx, Rx))

eq = [
    rho*(U*Ux + V*Uy) + Px - mu*(Uxx + Uyy) ~ 0,
    rho*(U*Vx + V*Vy) + Py - mu*(Vxx + Vyy) ~ 0,
    Ux + Vy ~ 0
    ]
bcs = [
    u(x,Ry) ~ v_slip,
    u(x,Ly) ~ 0,
    v(x,Ry) ~ 0,
    v(x,Ly) ~ 0,
    u(Lx,y) ~ 0,
    u(Rx,y) ~ 0,
    v(Lx,y) ~ 0,
    v(Rx,y) ~ 0,
    Dx(p(Lx, y)) ~ 0,
    Dx(p(Rx, y)) ~ 0,
    Dy(p(x, Ry)) ~ 0,
    Dy(p(x, Ly)) ~ 0,
    p(x, Ly) ~ 0,
    ]
# Space and time domains
domains = [
    x ∈ Interval(Lx, Rx),
    y ∈ Interval(Ly, Ry)
    ]

@named pdesys = PDESystem(
    eq, bcs, domains, 
    [x, y], 
    [u(x, y), v(x, y), p(x, y)]
    )

discretization = MOLFiniteDifference(
    [x => dx, y => dy], nothing, approx_order=order,
    advection_scheme = UpwindScheme())

prob = discretize(pdesys, discretization)
#@time sol = solve(prob,NewtonRaphson())
@time sol = solve(prob, NewtonRaphson(linsolve = KrylovJL_GMRES()))
#@time sol = solve(prob, KINSOL(;linear_solver = :GMRES))
u_num = sol[u(x, y)]
v_num = sol[v(x, y)]
p_num = sol[p(x, y)]
velocity_num = sqrt.(u_num.^2 + v_num.^2)

u_num = u_num'
v_num = v_num'
p_num = p_num'
velocity_num = velocity_num'

plot_nx = Int((Rx - Lx + dx)/dx)
plot_ny = Int((Ry - Ly + dy)/dy)
plot_x = range(Lx, Rx, length=plot_nx)
plot_y = range(Ly, Ry, length=plot_ny)

p1 = plot(plot_x, plot_y, u_num, linetype =:contourf, title = "u (m/s)", xlabel = "x (m)", ylabel = "y (m)", color =:turbo)
p2 = plot(plot_x, plot_y, v_num, linetype =:contourf, title = "v (m/s)", xlabel = "x (m)", ylabel = "y (m)", color =:turbo)
p3 = plot(plot_x, plot_y, velocity_num, linetype =:contourf, title = "velocity (m/s)", xlabel = "x (m)", ylabel = "y (m)", color =:turbo)
p4 = plot(plot_x, plot_y, p_num, linetype =:contourf, title = "p (Pa)", xlabel = "x (m)", ylabel = "y (m)", color =:turbo)
# 使用layout参数组合这些图形
combined_plot = plot(p1, p2, p3, p4, layout = (2, 2), size = (800, 600))
# 显示或保存图形
display(combined_plot)  # 显示图形