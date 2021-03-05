from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import imutils
import os
import shutil
import cv2
import numpy as np


def centroid_histogram(clt):
	# grab the number of different clusters and create a histogram
	# based on the number of pixels assigned to each cluster
	numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
	(hist, _) = np.histogram(clt.labels_, bins = numLabels)
	# normalize the histogram, such that it sums to one
	hist = hist.astype("float")
	hist /= hist.sum()
	# return the histogram
	return hist


def plot_colors(hist, centroids):
    # initialize the bar chart representing the relative frequency
    # of each of the colors
    bar = np.zeros((50, 300, 3), dtype="uint8")
    startX = 0
    # loop over the percentage of each cluster and the color of
    # each cluster
    for (percent, color) in zip(hist, centroids):
        # plot the relative percentage of each cluster
        endX = startX + (percent * 300)
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
                      color.astype("uint8").tolist(), -1)
        startX = endX

    # return the bar chart
    return bar


carcolors_lab={ 'red':[160,185],
           'purple':[130,160],
           'blue':[90,130],
           'green':[30,90],
           'yellow':[20,30],
           'brown':[0,20]
           }

def get_color(image_path):

    clusters=1

    image = cv2.imread(image_path)
    image=imutils.resize(image,height=800)
    h,w=image.shape[:2]
    image=image[int(0.2*h):int(0.8*h),int(0.2*w):int(0.8*w)]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = image.reshape((image.shape[0] * image.shape[1], 3))

    clt = KMeans(n_clusters = clusters)
    clt.fit(image)


    color_detected=clt.cluster_centers_[0]
    color_detected=[int(i) for i in color_detected]

    color_detected=np.array([[color_detected]],np.uint8)
    color_detected=cv2.cvtColor(color_detected,cv2.COLOR_RGB2HSV)
    color_detected=color_detected[0][0]

    color_res=0

    h_val=int(color_detected[0])
    s_val=int(color_detected[1])
    v_val=int(color_detected[2])

    if v_val<50:
        color_res='black'
    if s_val<40:
        if v_val<75:
            color_res='black'
        elif v_val >74 and v_val<130:
            color_res = 'gray'
        elif v_val >130 and v_val<180:
            color_res = 'silver'
        else:
            color_res='white'

    if color_res==0:
        for i in carcolors_lab.keys():
            if int(color_detected[0])<carcolors_lab[i][1] or int(color_detected[0])<carcolors_lab[i][1]>=carcolors_lab[i][0]:
                color_res=i
    return color_res

image_name='https:__s3.amazonaws.com_mc-imt_vehicle_2019A1250_front_right_view_29617_medium_image.jpg'
image_path='/home/gaurav/images/'+image_name

print('final color')
print(get_color(image_path))




