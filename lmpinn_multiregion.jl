using Flux
using Zygote
using NonlinearSolve, MINPACK
using Plots

# 定义模型
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

# 创建模型实例
input_dims = 2
output_dims = 1
num_layers = 1 # dims
num_neurals = 2 # orders
layers_neurals = fill(num_neurals, num_layers + 1)
model = f64(create_network(input_dims, output_dims, layers_neurals))
# 提取模型参数
flat, rebuild = Flux.destructure(model)
