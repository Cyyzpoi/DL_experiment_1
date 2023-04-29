#图像分类可视化
import numpy as np
import torchvision
import torch
import matplotlib.pyplot as plt
from data_process import test_loader
import matplotlib as mpl
print(mpl.get_backend())
mpl.use('TkAgg') #解决Pycharm中matplotlib报错问题

dict_={'0':"amusement",'1':"anger",'2':"awe",'3':"contentment",'4':"disgust",
       '5':"excitement",'6':"fear",'7':"sad"} #结果对应字典

Net=torch.load('./model/artphoto_model.pth')#加载模型

def imshow(img,title,ylabel):  #图像展示函数
    img=img.numpy().transpose((1,2,0))  #BGR转置
    mean=np.array([0.485,0.456,0.406])
    std=np.array([0.229,0.224,0.225])
    img=std*img+mean
    img=np.clip(img,0,1)
    plt.imshow(img)
    plt.ylabel('GroundTruth:{}'.format(ylabel))
    plt.title("predicted:{}".format(title))  #打印结果
    plt.show()



with torch.no_grad():  #取消梯度
    for data in test_loader:
        imgs,target=data
        target=target.cuda()
        out=torchvision.utils.make_grid(imgs)
        imgs = imgs.cuda()
        outputs=Net(imgs)
        _,predicted=torch.max(outputs.data,1)
        plt.figure()
        imshow(out,title=[dict_[str(predicted.item())]],ylabel=[dict_[str(target.item())]]) #打印图片
