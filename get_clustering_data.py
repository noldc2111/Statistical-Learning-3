from PIL import Image
from scipy import misc
from numpy import unique
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import csv,copy

#  This file is the the first in a pipeline to segment images.
#  Upon executuion, this script will read a training image and
#  a segmented training image, labeling the segmented image



def segment(im,image,v):
    """Take an image and a segmented images and 
       label the segmented image based on a unique value"""
    im_seg = copy.copy(im)
    for i in range(0,len(im[0])):
        for j in range(0,len(im[1])):
            if(image[i,j,0] == v):
                im_seg[i,j,] = [1,1,1]
            else:
                im_seg[i,j,] = [0,0,0]
    return im_seg

# read file
im = mpimg.imread("training_ss_151.png")
im_seg = mpimg.imread("training_seg_151.png")


im_seg_int = im_seg*255
v,c = unique(im_seg,return_counts=True)

im_back  = segment(im,im_seg,v[0])
im_csf   = segment(im,im_seg,v[1])
im_grey  = segment(im,im_seg,v[2])
im_white = segment(im,im_seg,v[3])

#plt.imshow(im_grey)
#plt.imshow(im*im_grey)
#plt.show()


d = []
index = 0
for i in range(0,len(im[0])):
    for j in range(0,len(im[1])):
        if(not im_back[i,j,1]):
            flat = [sub for each in im[(i-3):(i+2),(j-3):(j+2),1] for sub in each]
            d.append(flat)
            if(im_csf[i,j,1] == 1):
                d[index].append(1)
                d[index].append(i)
                d[index].append(j)
            if(im_grey[i,j,1] == 1):
                d[index].append(2)
                d[index].append(i)
                d[index].append(j)

            if(im_white[i,j,1] == 1):
                d[index].append(3)
                d[index].append(i)
                d[index].append(j)

            index += 1

im2 = im
print(im.shape)
im2[(61-2):(61+2),(109-2):(109+2),1]*255
im2[(61-2):(61+2),(109-2):(109+2),] = [1,1,1] 
#plt.imshow(im2)
#plt.show()

# write patches out to file
#with open("training_ss_151_blocks_python_labeled_coords.csv",'w') as csvfile:
#    datawriter = csv.writer(csvfile,delimiter=',')
#    for each in d:
#        datawriter.writerow(each)

