using Plots
function smoothStep(x; start=0.0, ends=10, steepness=10)
    # 使函数在x接近start时值接近0，在x接近end时值接近1
    normalized_x = (x - start) / (ends - start)
    return 1 / (1 + exp(-steepness * (normalized_x - 0.5)))
end
# 绘制函数和其一阶导数来验证
x_vals = 0:0.001:10
y_vals = [smoothStep(x) for x in x_vals]

plot(x_vals, y_vals, label="Smooth Step", title="Smooth Step Function and its Derivative")
