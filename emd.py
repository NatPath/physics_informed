import cv2
import numpy as np

def image_to_signature(image):
    x= image.shape[0]
    y= image.shape[1]
    signature=np.empty((image.size,3),dtype=np.float32)
    index=0
    for i in range(x):
        for j in range(y):
            signature[index]=np.array([image[i,j],i,j])
            index+=1
    return signature 

def emd(image1,image2):
    sig1=image_to_signature(image1)
    sig2=image_to_signature(image2)
    res=cv2.EMD(sig1,sig2,cv2.DIST_L2)
    return res[0]
