#将所给数据集进行分割，并存放至对应文件夹内
import re
import os
import cv2


def get_data(dir,txt):
    if os.path.exists("./data/"+dir)==False:
        os.makedirs("./data/"+dir)
    with open("./data/"+txt,'r') as f:
        data=f.read().splitlines() #获取文件名
        for root,dirs,files in os.walk("./data/testImages_artphoto"):  #获取路径
            for i in range(len(files)):
                if files[i] in data:
                    img=cv2.imread("./data/testImages_artphoto/"+files[i])
                    dir__=re.search(r'(.*?)_',files[i]) #利用正则表达式取出文件夹名即类名
                    if os.path.exists("./data/"+dir+"/"+dir__.group(1))==False:
                        os.makedirs("./data/"+dir+"/"+dir__.group(1))
                    cv2.imwrite("./data/"+dir+"/"+dir__.group(1)+"/"+files[i],img)#利用cv2完成图片数据的读取与保存
    print("the dataset is finished")



get_data("train","train.txt")
get_data("test","test.txt")  #分割数据集
