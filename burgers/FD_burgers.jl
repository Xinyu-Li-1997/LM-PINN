using ModelingToolkit
using ModelingToolkit: Interval, infimum, supremum
using MethodOfLines, DifferentialEquations
using Plots,CSV,DataFrames

@parameters x t
@variables u(..)

Dt = Differential(t)
Dx = Differential(x)
Dxx = Dx^2
α = 0.05
# Burger's equation
eq = Dt(u(t, x)) + u(t, x) * Dx(u(t, x)) - α * Dxx(u(t, x)) ~ 0

# boundary conditions
bcs = [
    u(0.0, x) ~ -sin(π * x),
    u(t, -1.01) ~ 0.0,
    u(t, 1.01) ~ 0.0
]

domains = [t ∈ Interval(0.0, 0.99), x ∈ Interval(-1.01, 1.01)]

# MethodOfLines, for FD solution
dx = 0.02
discretization = MOLFiniteDifference([x => dx], t, saveat = 0.01)
@named pde_system = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])
prob = discretize(pde_system, discretization)
@time sol = solve(prob, KenCarp47(linsolve = KrylovJL_GMRES()), save_everystep = false)
ts = sol[t]
xs = sol[x]
u_MOL = sol[u(t, x)]
pp = plot(ts, xs, u_MOL', linetype =:contourf, size = (800,600))
display(pp)
# 写入CSV文件
out = u_MOL'[2:101,:]
CSV.write("Burgers$(α).csv", DataFrame(out,:auto))