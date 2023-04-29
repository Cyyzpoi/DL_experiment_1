import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from data_process import data_process
from torch.utils.tensorboard import SummaryWriter
# Net=torchvision.models.vit_b_16(weights=torchvision.models.ViT_B_16_Weights)
Net=torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
# Net=torchvision.models.resnet34(weights=torchvision.models.ResNet34_Weights.IMAGENET1K_V1)
# Net=torchvision.models.resnet101(weights=torchvision.models.ResNet101_Weights.IMAGENET1K_V2)

#读取torchvision中预训练模型

# Net=torch.load('model/artphoto_model.pth')
# for params in Net.parameters():
#     params.requires_grad=True

#读取现有模型

# Net.layer1.append(nn.Dropout(0.2))
# Net.layer2.append(nn.Dropout(0.4))
# Net.layer3.append(nn.Dropout(0.2))
Net.layer4.append(nn.Dropout(0.3))

#对Resnet50网络添加Dropout

for params in Net.parameters():
    params.requires_grad=False

#调用预训练模型时将参数冻结

# print(Net)
Net.fc=nn.Sequential(nn.Linear(in_features=2048,out_features=8)).cuda()
Net=Net.cuda()
#更改全连接层使之适应八分类任务
# print(Net)

acc=0.0
train_loader,test_loader=data_process()
lr=0.01
epoch=50
batch_size=64
gpu=torch.cuda.is_available()
optimizer=optim.Adam(Net.parameters(),lr=lr,weight_decay=5e-4)
writer=SummaryWriter(log_dir='logs/Resnet50+tiny_Dropout_{}'.format(time.strftime('%Y%m%d-%H%M%S')))

if gpu:
    loss_fn=nn.CrossEntropyLoss().cuda()
else:
    loss_fn=nn.CrossEntropyLoss()

train_data_size=len(train_loader)
test_data_size=len(test_loader)

#定义各超参数，优化器，损失函数，并利用tensorboard进行数据可视化


#训练函数
def train(epoch):
        train_loss = 0
        correct = 0
        total = 0
        for index, (imgs, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            if gpu:
                imgs, targets = imgs.cuda(), targets.cuda()
            output = Net(imgs)
            loss = loss_fn(output, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(output, dim=1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

            if (index + 1) % 50 == 0 or index == (len(train_loader) - 1):
                print("epoch:{},loss:{:.6f}| ACC:{:.6f}% ({}/{})".format(epoch, train_loss / (index + 1), 100 * correct / total,
                                                                 correct, total))
        writer.add_scalar('train_loss',loss.item(),epoch)
        writer.add_scalar('train_acc',100*correct/total,epoch)
#测试函数
def test_acc():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            imgs, targets = data
            if gpu:
                imgs, targets = imgs.cuda(), targets.cuda()
            outputs = Net(imgs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    print('Test Accuracy: {}/{} ({:.0f}%)'.format(correct, total, 100. * correct / total))
    return correct / total

import os


if __name__ == '__main__': #主函数部分
    for i in range(1,epoch+1):
        print("-----------------Epoch: {}-----------------".format(i))
        train(i)
        acc1 = test_acc()
        writer.add_scalar('test_acc',acc1,i)  #存取测试准确率
        if os.path.exists('./model') is not True:
            os.mkdir("./model")
        if acc1 > acc:
            acc = acc1
            torch.save(Net, 'model/artphoto_model_'+str(acc)[0:5]+'.pth')#保存模型
            print('Saved model')

    writer.close()