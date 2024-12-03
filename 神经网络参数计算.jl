function calculate_network_parameters(Ni, Nhs, No)
    if length(Nhs) == 1
        total_params = 2 * (Ni * Nhs[1] + Nhs[1] * No)
    else
        total_params = 2 * (Ni * Nhs[1] + sum(Nhs[i] * Nhs[i+1] for i in 1:length(Nhs)-1) + Nhs[end] * No)
    end
    return total_params
end
#=
N_degree = 2 # 配点多项式阶数
Ni = 4  # 输入层神经元数量 -- 对应坐标数量/维度
Nhs = [N_degree].+ zeros(Int, Ni)  # 各隐含层神经元数量 -- 对应多项式等效阶数
No = 1  # 输出层神经元数量 -- 对应解数量
Np = (N_degree + 1)^Ni # 配点数 -- 对应约束个数
params = calculate_network_parameters(Ni, Nhs, No)
println("神经网络参数数量为: ", params)
println("时空配点数量为: ", Np)
=#
# Iterate over Ni and N_degree values from 1 to 4, calculate and print params and Np with labels
No = 3 # 输出层神经元数量 -- 对应解数量
Ni_max = 4 # 输入层神经元数量 -- 对应坐标数量/维度
N_degree_max = 4 # 配点多项式阶数
for Ni in 1:Ni_max
    for N_degree in 1:N_degree_max
        Nhs = [N_degree].+ zeros(Int, Ni)
        Np = (N_degree + 1)^Ni*No
        params = calculate_network_parameters(Ni, Nhs, No)
        print("Dim = $Ni,Degree = $N_degree: \tN_nn = $params,N_loss = $Np\t")
    end
    println()
end
