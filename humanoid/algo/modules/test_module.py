import torch
from cnn import CNN
from mlp import MLP  # 如果是测试 CNN，请替换为 CNN

def test_mlp():
    # 模型参数
    input_dim = 512
    output_dim = 3
    hidden_dims = [256, 128]

    # 创建模型
    model = MLP(input_dim=input_dim, output_dim=output_dim, hidden_dims=hidden_dims, activation='elu')

    # 构造测试输入 (batch_size, input_dim)
    batch_size = 4
    x = torch.randn(batch_size, input_dim)

    # 前向传播
    y = model(x)

    # 打印形状
    print("Input shape:", x.shape)
    print("Output shape:", y.shape)
    print("Output values:", y)

def test_cnn_module():
    print("===== Testing CNN Module =====")

    # 1️⃣ 创建一个 CNN 实例
    cnn = CNN(
        input_channels=1,
        input_size=[14, 24],
        conv_channels=[16, 32, 64],
        kernel_sizes=[3, 3, 3],
        strides=[2, 2, 2],
        paddings=[1, 1, 1],
        activation='elu',
        output_dim=128,
        name='test_cnn'
    )

    # 2️⃣ 测试输入
    batch_size = 8
    dummy_input = torch.randn(batch_size, 1, 14, 24)

    # 3️⃣ 前向传播
    output = cnn(dummy_input)

    # 4️⃣ 打印结构和输出结果
    print("\n--- Model structure ---")
    print(cnn)
    print("\n--- Input shape ---")
    print(dummy_input.shape)
    print("\n--- Output shape ---")
    print(output.shape)

    # 5️⃣ 简单检查维度
    assert output.shape == (batch_size, 128), f"Unexpected output shape: {output.shape}"
    print("\n✅ CNN forward pass successful!")

# 直接运行测试
if __name__ == "__main__":
    test_cnn_module()
    test_mlp()
