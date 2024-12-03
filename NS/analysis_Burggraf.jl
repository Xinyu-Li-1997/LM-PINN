using ForwardDiff
# 定义前向微分变量
fX(x) = x^4-2*x^3+x^2
dfX(x) = ForwardDiff.derivative(fX, x)
ddfX(x) = ForwardDiff.derivative(dfX, x)
dddfX(x) = ForwardDiff.derivative(ddfX, x)
gY(y) = y^4-y^2
dGY(y) = ForwardDiff.derivative(gY, y)
ddGY(y) = ForwardDiff.derivative(dGY, y)
dddGY(y) = ForwardDiff.derivative(ddGY, y)
ddddGY(y) = ForwardDiff.derivative(dddGY, y)
Fx(x) = x^5/5-x^4/2+x^3/3
F1x(x) = -4*x^6+12*x^5-14*x^4+8*x^3-2*x^2
F2x(x) = fX(x)^2/2
G1y(y) = -24*y^5+8*y^3-4*y

fex(x,y,Re) = 8/Re*(24*Fx(x)+2*dfX(x)*ddGY(y)+dddfX(x)*gY(y))+64*(F2x(x)*G1y(y)-gY(y)*dGY(y)*F1x(x))
U_slipx(x) = 16*(x^4-2*x^3+x^2)

u_ana(x,y) = 8*(x^4-2*x^3+x^2)*(4*y^3-2*y)
v_ana(x,y) = -8*(4*x^3-6*x^2+2*x)*(y^4-y^2)
p_ana(x,y,Re) = 8/Re*(Fx(x)*dddGY(y) + dfX(x)*dGY(y)) + 64*F2x(x)*(gY(y)*ddGY(y) - dGY(y)^2)

#=
using Plots
# 计算域设定
Re = 10.0
Lx = 0.0
Rx = 1.0
Ly = 0.0
Ry = 1.0
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
        u_num[i, j] = u_ana(plot_x[i], plot_y[j])
        v_num[i, j] = v_ana(plot_x[i], plot_y[j])
        velocity_num[i, j] = (u_num[i, j]^2 + v_num[i, j]^2)^0.5
        p_num[i, j] = p_ana(plot_x[i], plot_y[j], Re)
    end
end
u_num = u_num'
v_num = v_num'
velocity_num = velocity_num'
p_num = p_num'
p1 = contourf(plot_x, plot_y, u_num, lw = 0, levels=20, title = "u (m/s)", xlabel = "x (m)", ylabel = "y (m)", color =:turbo)
p2 = contourf(plot_x, plot_y, v_num, lw = 0, levels=20, title = "v (m/s)", xlabel = "x (m)", ylabel = "y (m)", color =:turbo)
p3 = contourf(plot_x, plot_y, velocity_num, lw = 0, levels=20, title = "velocity (m/s)", xlabel = "x (m)", ylabel = "y (m)", color =:turbo)
p4 = contourf(plot_x, plot_y, p_num, lw = 0, levels=20, title = "p (Pa)", xlabel = "x (m)", ylabel = "y (m)", color =:turbo)
# 使用layout参数组合这些图形
combined_plot = plot(p1, p2, p3, p4, layout = (2, 2), size = (800, 675))
# 显示或保存图形
display(combined_plot)  # 显示图形
=#