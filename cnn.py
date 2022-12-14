#_____ ISSUES______


from email.mime import image
import os

from glob import glob
import pandas as pd
import cv2

from os import listdir
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import binascii
import random 
import shutil
import splitfolders
import leargist


# Step 1:PreProcessing (Convert Binary Files to Images) + resizing them to (64X64)

# num_files = listdir("ml-sample-pack-small/malware/arm")

# Img=[]
# size_files = []

# # for i in num_files:
# try:
#     with open("ml-sample-pack-small/malware/arm/" + num_files[1], "rb") as f:
#         file_size = os.path.getsize("ml-sample-pack-small/malware/arm/" + num_files[0])
#         file_size_kb = file_size/1024
#         print(file_size_kb)
#         if(file_size_kb<10):
#             width =32
#         elif(file_size_kb>=10 and file_size_kb<30):
#             width =64
#         elif(file_size_kb>=30 and file_size_kb<60):
#             width =128
#         elif(file_size_kb>=60 and file_size_kb<100):
#             width =256
#         elif(file_size_kb>=100 and file_size_kb<200):
#             width =384
#         elif(file_size_kb>=200 and file_size_kb<500):
#             width =512
#         elif(file_size_kb>=500 and file_size_kb<1000):
#             width = 768
#         else:
#             width = 1024
#         numpy_1d_data = np.fromfile(f,dtype)
#         #print(numpy_1d_data)
#         numpy_2d_data  = []
#         #print(width)
#         for j in range(0,len(numpy_1d_data),1):
#             temp = []
#             for m in range(0,width,1):
#                 temp.append(numpy_1d_data[m])
#             numpy_2d_data.append(temp)
#         #print(numpy_2d_data)
#         numpy_2d_data = np.array(numpy_2d_data) 
#         #print(type(numpy_2d_data))  
#         #print(numpy_2d_data)
#         im = Image.fromarray(numpy_2d_data)
#         #im = im.resize((64,64))
#         # # to view what's wrong!!
#         imgplot = plt.imshow(im)
#         plt.show()
#             # Img.append(im)
# except IOError:
#     print('Error While Opening the file!')  


images_new = []
def start():
    num_files = listdir("ml-sample-pack-small/benign/arm")
    for file in num_files:
        loc = "ml-sample-pack-small/benign/arm/" + file
        with open(loc,'rb')as f:
            content=f.read()
        hexst=binascii.hexlify(content)
        fh=np.array([int(hexst[i:i+2],16)for i in range(0,len(hexst),2)])
        rn=len(fh)/1024
        fh=np.reshape(fh[:int(rn)*1024],(-1,1024))
    #   print(fh.shape)
        fh=np.uint8(fh)
        im=Image.fromarray(fh)
        im=im.resize((32,32),Image.ANTIALIAS)
        images_new.append(im)
        # im.show()
    
    # imgplot = plt.imshow(im)
    # plt.show()

def saving():
    image_path=os.path.join("/Users/chaitanya/Desktop/Training Project/","Images-bengin")
    os.mkdir(image_path)
    for i in images_new:
        i.save(image_path +'/'+ str(i)+".png")
        print("save image:"+image_path+".png")


def rename():
    num_files = listdir("Images-arm-malware")
    m=1
    for file in num_files:
        os.rename("Images-arm-malware/"+str(file),"Images-arm-malware/"+ "Images-arm-malware"+str(m)+".png")
        m+=1


def preparation():
    num_files = listdir("Dataset/Images-arm-malware")
    # print(type(num_files[0]))

    for file in num_files:
        im=Image.open("Dataset/Images-arm-malware/" + file)
        im=im.resize((64,64))
        im.save("Dataset/Images-arm-malware/" + file)

    num_files2 = listdir("Dataset/Images-arm-bengin")
    for file1 in num_files2:
        im=Image.open("Dataset/Images-arm-bengin/" + file1)
        im=im.resize((64,64))
        im.save("Dataset/Images-arm-bengin/" + file1)

    splitfolders.ratio("Dataset/", output="Dataset_new", 
                   seed=42, ratio=(.8,.2),
                   group_prefix=None) 

    ## label the dataset for both
    ## combine the two training and test dataset of malware and Bengin
    ## Randomize them


def labelling():
    X = []
    df = pd.DataFrame(columns = ['image','label'])
    y = []
    for file in glob('./Dataset/Images-arm-bengin/.*png'):
        img = cv2.imread(file)
        if file is not None:
            img_arr = np.asarray(img)
            X.append(img_arr)
            y.append(1)
    X = np.asarray(X)
    y = np.asarray(y)
    print(to_categorical(y,num_classes = 2))




def trying():
    img = Image.open("Dataset/Images-arm-bengin/Images-arm-bengin1.png")
    # x = np.asarray(img)
    # descriptor = gist.extract(x)
    descriptor = leargist.color_gist(img)
    print(descriptor)


start()
# print(type(images_new[0]))
# saving()
# rename()
# preparation()

# labelling()
# trying()
