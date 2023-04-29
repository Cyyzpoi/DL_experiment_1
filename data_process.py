#数据预处理与加载文件
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

transform_train=transforms.Compose([transforms.RandomRotation(45),
                              transforms.CenterCrop((224,224)),
                              transforms.RandomHorizontalFlip(0.5),
                              transforms.ColorJitter(brightness=0.2,contrast=0.1,saturation=0.1,hue=0.2),
                              transforms.ToTensor(),
                              transforms.Normalize(([0.485, 0.456, 0.406]),(0.229, 0.224,
0.225))])   #训练集数据预处理部分，包括数据增广等

transform_test=transforms.Compose([transforms.Resize((256,256)),
                                   transforms.CenterCrop((224,224)),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224,
0.225))])   #测试集数据预处理，为提高精准度去除了随机翻转等

def data_process():
    train_data = datasets.ImageFolder(root="./data/train", transform=transform_train)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_data=datasets.ImageFolder(root="./data/test",transform=transform_test)
    test_loader=DataLoader(test_data,batch_size=1,shuffle=False)

    return train_loader,test_loader

train_loader,test_loader=data_process()

#加载数据，选择batchsize等参数


