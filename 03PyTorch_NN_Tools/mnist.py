import matplotlib
import numpy as np
import torch

# 导入pytorch内置的mnist数据
from torchvision.datasets import mnist

# 导入预处理模块
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# 导入nn及优化器
import torch.nn.functional as F
import torch.optim as optim
from torch import nn

# 2.定义一些超参数
train_batch_size = 64
test_batch_size = 64
learning_rate = 0.01
num_epoches = 20
lr = 0.01
momentum = 0.5

# 3.下载数据预处理
# 定义预处理函数，这些预处理依次放在Compose函数中。
# 1）transforms.Compose可以把一些转换函数组会在一起
# 2）Normalize([0,0.5],[0,0.5])对张量归一化，两个0.5分别表示对张量进行归一化的全局平均值核方差。
# 3）download参数控制是否需要下载，若./data目录下已有MNIST，选择False
# 4）用DataLoader得到生成器看，节省内存。
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
#  下载数据，并对数据做预处理
train_dataset = mnist.MNIST('./data', train=True, transform=transform, download=True)
test_dataset = mnist.MNIST('./data', train=False, transform=transform)
# dataloader是一个可迭代对象，可以使用迭代器一样使用。
train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

# 3.可视化源数据
import matplotlib.pyplot as plt

examples = enumerate(test_loader)
batch_idx, (examples_data, examples_targets) = next(examples)

fig = plt.figure()
for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.tight_layout()
    plt.imshow(examples_data[i][0], cmap='gray', interpolation='none')
    plt.title('Groud Truth:{}'.format(examples_targets[i]))
    plt.xticks([])
    plt.yticks([])
    # plt.show()


# 4.构建模型
# 4.1 构建网络
class Net(nn.Module):
    """
    使用sequential构建网络，Sequential()函数功能将网络的层组合到一起
    """
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(Net, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden_1), nn.BatchNorm1d(n_hidden_1))
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2), nn.BatchNorm1d(n_hidden_2))
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, out_dim))

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x


# 4.2 实例化网络
# 检测是否有可用GPU，有则使用，否则使用CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 实例化网络
model = Net(28 * 28, 300, 100, 10)
model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

# 5.训练模型
# 包括对训练数据的训练模型，然后用测试数据的验证模型
# 5.1 训练模型
losses = []
acces = []
eval_losses = []
eval_acces = []

for epoch in range(num_epoches):
    train_loss = 0
    train_acc = 0
    # 使模型处于训练模式，会把所有的module设置为训练模式。
    model.train()

    # 动态修改参数学习率
    if epoch % 5 == 0:
        optimizer.param_groups[0]["lr"] = lr * 0.1

    for img, label in train_loader:
        img = img.to(device)
        label = label.to(device)
        img = img.view(img.size(0), -1)
        # 前向传播
        # （1）正向传播生成网络的输出，
        out = model(img)
        # （2）计算输出和实际值之间的损失值
        loss = criterion(out, label)

        # 反向传播
        # 缺省情况下梯度是累加的，在梯度反向传播前，需要手工把梯度初始化或清零。
        optimizer.zero_grad()
        # 自动生成梯度
        loss.backward()
        # 更新参数：执行优化器，基于当前梯度更新参数，把梯度传播回每个网络
        optimizer.step()
        train_loss += loss.item()
        # 计算分类的准确率
        _, pred = out.max(1)
        num_correct = (pred == label).sum().item()
        acc = num_correct / img.shape[0]
        train_acc += acc
    losses.append(train_loss / len(train_loader))
    acces.append(train_acc / len(train_loader))

    # 在测试集上验证效果
    eval_loss = 0
    eval_acc = 0
    # 将模型改为预测模式
    model.eval()
    for img, label in test_loader:
        img = img.to(device)
        label = label.to(device)
        img = img.view(img.size(0), -1)
        out = model(img)
        loss = criterion(out, label)
        # 记录误差
        eval_loss += loss.item()
        # 记录准确率
        _, pred = out.max(1)
        num_correct = (pred == label).sum().item()
        acc = num_correct / img.shape[0]
        eval_acc += acc
    eval_losses.append(eval_loss / len(test_loader))
    eval_acces.append(eval_acc / len(test_loader))
    print('epoch: {},Train Loss:{:.4f},Train Acc: {:.4f} ,Test Loss:{:.4f},Test Acc:{:.4f}'
          .format(epoch, train_loss / len(train_loader), train_acc / len(train_loader), eval_loss / len(test_loader),
                  eval_acc / len(test_loader)))
# 5.2 可视化训练及测试损失值
plt.title('trainloss')
plt.plot(np.arange(len(losses)),losses)
plt.legend(['Train Loss'],loc='upper right')
