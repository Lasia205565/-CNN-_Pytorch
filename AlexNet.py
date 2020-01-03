import torch
import torchvision
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#使用compose函数来将多个转换器结合在一起

# 第一步：加载训练数据和测试数据

trainset = torchvision.datasets.ImageFolder(
        "/home2/lthpc/datasets/ImageNet/train",
        transforms.Compose(
            [transforms.RandomResizedCrop(224),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(), normalize]))

testLoader = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder("/home2/lthpc/datasets/ImageNet/val", transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=128, shuffle=False,
        num_workers=4, pin_memory=True)

trainLoader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)

#第二步：定义CNN网络

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        #(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size = 11, stride=4,padding=0) # batch, channel , height , width
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

        self.conv2 = nn.Conv2d(in_channels=96,out_channels=256,kernel_size= 5,stride=1,padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size = 3,stride=2,padding=0)

        self.conv3 = nn.Conv2d(in_channels = 256,out_channels = 384,kernel_size=3,stride=1,padding=1)

        self.conv4 = nn.Conv2d(in_channels=384,out_channels=384,kernel_size=3,stride=1,padding=1)

        self.conv5 = nn.Conv2d(in_channels=384,out_channels=256,kernel_size=2,stride=1,padding=1)
        self.pool5 = nn.MaxPool2d(kernel_size=3,stride=2,padding=0)

        self.fc1 = nn.Linear(6*6*256, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 1000)


    def forward(self, x):
         x = F.relu(self.conv1(x))
         x = self.pool1(x)

         x = F.relu(self.conv2(x))
         x = self.pool2(x)

         x = F.relu(self.conv3(x))

         x = F.relu(self.conv4(x))

         x = F.relu(self.conv5(x))
         x = self.pool5(x)
         
         x = x.view(x.size(0), -1)

         x= F.relu(F.dropout(self.fc1(x), 0.5,training=self.training))
         x= F.relu(F.dropout(self.fc2(x), 0.5,training=self.training))


         x = self.fc3(x)
         return x


net = Net()
net.to(device)
# 第三步：定义损失函数和优化器

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

#第四步：训练神经网络
for epoch in range(100):
    running_loss =0.0
    #其中data是列表[inputs,labels]
    total = 0
    correct = 0
    for i, data in enumerate(trainLoader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        print('Epoach:%d loss:%.3f  Accuracy:%.3f %%' % (epoch+1, running_loss/(i+1), 100 * correct / total))


    if(epoch%5==1):
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testLoader:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print('Epoach:%d Accuracy of network during test:%d %%' % (epoch+1, 100 * correct / total))



print('Finish Training')

#pro1:保存训练好的模型
#PATH = '/cifar_net.pth'
#torch.save(net.state.dict(),PATH)

#第五步：测试数据

#加载一个batch_size的数据用于测试
#dataiter = iter(testLoader)
#images,labels = dataiter.next()

#显示测试图像
#plt.imshow(torchvision.utils.make_grid(images))
#显示测试图像对应的标签，其中join是连接两个字符串，labels[j]--第j的测试样本对应的种类索引，classes[labels[j]]表示索引对应的种类
#print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(10)))


#pro2:调用保存好的模型
#net = Net()
#net.load_state_dict(torch.load(PATH))
#outputs = net(images)



#torch.max(a,0) 返回每一列中最大的那个元素，且返回索引
#torch.man(a,1) 返回每一行中最大的那个元素，且返回索引
#_,predicted = torch.max(outputs,1)
#print('Predicted:',' '.join('%5s' % classes[predicted[j]] for j in range(10)))


