# 1)导入模块
import torch
import torch.utils.data as Data
import torch.nn.functional as F
import matplotlib.pyplot as plt
from Cython import inline
# 超参数
lr = 0.01
batch_size = 32
num_epoches = 12

# 2) 生成数据
# 生成训练数据
# torch.unsqueeze() # 一维变为二维，torch只能处理二维的数据
x = torch.unsqueeze(torch.linspace(-1, 1, 1000), dim=1)
# 0.1*torch.normal(x.size()) # 增加噪点
y = x.pow(2) + 0.1 * torch.normal(torch.zeros(*x.size()))
torch_dataset = Data.TensorDataset(x, y)
# 得到一个代批量的生成器
loader = Data.DataLoader(dataset=torch_dataset, batch_size=batch_size, shuffle=True)


# 3)构建神经网络
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(1, 20)
        self.predict = torch.nn.Linear(20, 1)

    # 前向传递
    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x


# 4)使用多种优化器
net_SGD = Net()
net_Momentum = Net()
net_RMSProp = Net()
net_Adam = Net()

nets = [net_SGD, net_Momentum, net_RMSProp, net_Adam]
opt_SGD = torch.optim.SGD(net_SGD.parameters(), lr=lr)
opt_Momentum = torch.optim.SGD(net_Momentum.parameters(), lr=lr, momentum=0.9)
opt_RMSProp = torch.optim.RMSprop(net_RMSProp.parameters(), lr=lr, alpha=0.9)
opt_Adam = torch.optim.Adam(net_Adam.parameters(), lr=lr, betas=(0.9, 0.99))
optimizers = [opt_SGD, opt_Momentum, opt_RMSProp, opt_Adam]

# 5)训练模型
loss_func = torch.nn.MSELoss()
loss_his = [[], [], [], []]  # 记录损失
for epoch in range(num_epoches):
    for step, (batch_x, batch_y) in enumerate(loader):
        for net, opt, l_his in zip(nets, optimizers, loss_his):
            output = net(batch_x)  # get output for every net
            loss = loss_func(output, batch_y)  # compute loss for every net
            opt.zero_grad()  # clear gradinet for net train
            loss.backward()  # backpropagation,compute gradients
            opt.step()  # apply gradients
            l_his.append(loss.data.numpy())  # loss recoder

    labels = ['SGD', 'Momentum', 'RMSprop', 'Adam']

# 6)可视化结果
for i, l_his in enumerate(loss_his):
    plt.plot(l_his, label=labels[i])
    plt.legend(loc='best')
    plt.xlabel(' Steps')
    plt.ylabel(' Loss')
    plt.ylim((0, 0.2))
    plt.show()
