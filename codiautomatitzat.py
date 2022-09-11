#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 00:04:44 2022

@author: xavi
"""

global k

import cv2
import pandas as pd
import numpy as np
import numpy.linalg 
import matplotlib.pyplot as plt
from astroML.plotting import scatter_contour
from skimage import data
from skimage import filters
from skimage import exposure
from scipy import ndimage
from scipy.signal import argrelextrema

from shapely import wkt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from PIL import Image

from raw import *
from utils import *
from birdeyeview_lidar import *



##%%

#we load the class of the raw data
data = raw('/home/xavi/TFM/data/','2011_09_26','0027')
#data = raw('/home/xavi/TFM/data/','2011_09_30','0016')

oxtsdata = data.oxts_files
poses = load_oxts_packets_and_poses(oxtsdata)

#%%

dataoxts = pd.read_csv('/home/xavi/TFM/data/2011_09_26/2011_09_26_drive_0027_sync/oxts/data/0000000000.txt', sep=" ", header=None)
#dataoxts = pd.read_csv('/home/xavi/TFM/data/2011_09_30/2011_09_30_drive_0016_sync/oxts/data/0000000000.txt', sep=" ", header=None)
dataoxts.columns = ['lat', 'lon', 'alt',
                        'roll', 'pitch', 'yaw',
                        'vn', 've', 'vf', 'vl', 'vu',
                        'ax', 'ay', 'az', 'af', 'al', 'au',
                        'wx', 'wy', 'wz', 'wf', 'wl', 'wu',
                        'pos_accuracy', 'vel_accuracy',
                        'navstat', 'numsats',
                        'posmode', 'velmode', 'orimode']


timestamps = pd.read_csv('/home/xavi/TFM/data/2011_09_26/2011_09_26_drive_0027_sync/oxts/timestamps.txt', sep=" |:", header=None)
#timestamps = pd.read_csv('/home/xavi/TFM/data/2011_09_30/2011_09_30_drive_0016_sync/oxts/timestamps.txt', sep=" |:", header=None)
timestamps.columns = ['date','hh','mm','ss']

#%%
"""we load the data from the oxts sensor"""

for ii1 in range(1,len(timestamps)):
    oxtnumstr = (str(ii1).zfill(3))
    dataoxtsi = pd.read_csv('/home/xavi/TFM/data/2011_09_26/2011_09_26_drive_0027_sync/oxts/data/0000000'+r'{}'.format(oxtnumstr)+r".txt", sep=" ", header=None)
    #dataoxtsi = pd.read_csv('/home/xavi/TFM/data/2011_09_30/2011_09_30_drive_0016_sync/oxts/data/000000'+r'{}'.format(oxtnumstr)+r".txt", sep=" ", header=None)
    dataoxtsi.columns = ['lat', 'lon', 'alt',
                        'roll', 'pitch', 'yaw',
                        'vn', 've', 'vf', 'vl', 'vu',
                        'ax', 'ay', 'az', 'af', 'al', 'au',
                        'wx', 'wy', 'wz', 'wf', 'wl', 'wu',
                        'pos_accuracy', 'vel_accuracy',
                        'navstat', 'numsats',
                        'posmode', 'velmode', 'orimode']

    dataoxts = pd.concat([dataoxts,dataoxtsi], ignore_index=True, axis=0)




#%%

def getGround_World(points, THRESHOLD):
    
    xyz = points
    height_col = int(np.argmin(np.var(xyz[:,:3], axis = 0)))
    
    # temp = np.zeros((len(xyz[:,1]),4), dtype= float)
    # temp[:,:3] = xyz[:,:3]
    # temp[:,3] = np.arange(len(xyz[:,1]))
    # xyz = temp
    z_filter = xyz[xyz[:,height_col] < np.float32(-1.3)]
    #z_filter = xyz[(xyz[:,height_col]< np.mean(xyz[:,height_col]) + 10.0*np.std(xyz[:,height_col])) & (xyz[:,height_col]> np.mean(xyz[:,height_col]) - 1.5*np.std(xyz[:,height_col]))]
    
    max_z, min_z = np.max(z_filter[:,height_col]), np.min(z_filter[:,height_col])
    z_filter[:,height_col] = (z_filter[:,height_col] - min_z)/(max_z - min_z)
    iter_cycle = 10
    
    
    for i in range(iter_cycle):
        covariance = np.cov(z_filter[:,:3].T)
        w,v,h = np.linalg.svd(np.matrix(covariance))
        normal_vector = w[np.argmin(v)]
        filter_mask = np.asarray(np.abs(np.matrix(normal_vector)*np.matrix(z_filter[:,:3]).T )<THRESHOLD)
        z_filter = np.asarray([z_filter[index[1]] for index,a in np.ndenumerate(filter_mask) if a == True])
        
        z_filter[:,height_col] = z_filter[:,height_col]*(max_z - min_z) + min_z
        
       
        
        
      
        return z_filter
    
    

def filter_by_mean_value(pointcloud_df, left_bound, right_bound):

    mean = pointcloud_df[:,3].mean()
    std = pointcloud_df[:,3].std()

    lanes_df = pointcloud_df[pointcloud_df[:,3] > mean + left_bound * std]
    lanes_df = lanes_df[lanes_df[:,3] < mean + right_bound * std ]

    print("MEAN FILTER:")
    print("============")
    print("Intensity - Mean value:      ", mean)
    print("Intensity - Std value:       ", std)
    print("Intensity - Lower bound:     ", mean + 1 * std)
    print("Intensity - Upper bound:     ", mean + 7 * std)
    print("Intensity - Filtered points: ", len(lanes_df))
    print("Intensity - Original points: ", len(pointcloud_df))
    print("Intensity - Reduction to %:  ", len(lanes_df)/len(pointcloud_df))
    
    return lanes_df






#%%


 



for ii in range(10,172):
    dataveloaug = data.get_velo(ii) #we get the 3D LiDAR pointcloud for the frame
    dataveloaugrot = data.get_velo(ii)
    dataveloaugrot = getGround_World(dataveloaugrot,3.0)
    
    poseii = (poses[ii][1])[:,3]
    poserotii = (poses[ii][1])[0:3,0:3]
    poserotii2 = np.copy(poserotii[0:3,0:3])
    poserotii2[0,0] = -poserotii[0,1]
    poserotii2[0,1] = -poserotii[0,0]
    poserotii2[1,0] = poserotii[0,0]
    poserotii2[1,1] = -poserotii[0,1]
    
 
    for jj in [x for x in range(-7,8) if x!=0]: #we iterate for the 10 previous and posterior frames
        dataveloaugii = data.get_velo(ii+jj)
        dataveloaugii = getGround_World(dataveloaugii,3.0)
        dataveloaugiirot = data.get_velo(ii+jj)
        dataveloaugiirot = getGround_World(dataveloaugiirot,3.0)
        dataveloaugiirot2 = data.get_velo(ii+jj)
        
        posejj = (poses[ii+jj][1])[:,3]
        poserotjj = (poses[ii+jj][1])[0:3,0:3]
        poserotjj2 = np.copy(poserotjj[0:3,0:3])
        poserotjj2[0,0] = -poserotjj[0,1]
        poserotjj2[0,1] = -poserotjj[0,0]
        poserotjj2[1,0] = poserotjj[0,0]
        poserotjj2[1,1] = -poserotjj[0,1]
        
        
        
        cosangleii = -(poses[ii][1])[1,0]
        cosanglejj = -(poses[ii+jj][1])[1,0]
        sinangleii = (poses[ii][1])[0,0]
        sinanglejj = (poses[ii+jj][1])[0,0]
        
        
        
        dataveloaugii[:,0] = dataveloaugii[:,0]-((posejj[1]-poseii[1])*0.5*(cosangleii+cosanglejj)-(posejj[0]-poseii[0])*0.5*(sinangleii+sinanglejj))
        dataveloaugii[:,1] = dataveloaugii[:,1]-((posejj[1]-poseii[1])*0.5*(sinangleii+sinanglejj)+(posejj[0]-poseii[0])*0.5*(cosangleii+cosanglejj))
        dataveloaugii[:,2] = dataveloaugii[:,2]-(posejj[2]-poseii[2])
        
        #dataveloaugiirot[0:3,0:3] = np.dot(dataveloaugii[0:3,0:3],poserotjj)
        #dataveloaugiirot2[0:3,0:3] = np.dot(dataveloaugiirot[0:3,0:3],(poserotii.transpose()))
        
        poserotiijj = np.dot(poserotii2.transpose(),poserotjj2)
        dataveloaugiirot[:,0:3] = np.dot(dataveloaugii[:,0:3],poserotiijj)
        
        

        dataveloaugrot = np.concatenate((dataveloaugrot,dataveloaugiirot),axis=0)
 
        


    lidar = dataveloaugrot[:,0:4]

    lidar2 = lidar[np.logical_not(np.logical_or(lidar[:,1] < -10.0, lidar[:,1] > 10.0))]

    lidar_mean_filter = filter_by_mean_value(lidar2, 2.5, 40.0)
    
    
    
    
    X = StandardScaler().fit_transform(lidar_mean_filter[:,:3])
    
    
    db = DBSCAN(eps=0.06, min_samples=10).fit(X)
    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    
    lidar_mean_filter = np.c_[lidar_mean_filter,labels]
    
    lidar_mean_filter_db = lidar_mean_filter[np.logical_not(lidar_mean_filter[:,4] < 0)]
    
    lidar_repr = birds_eye_point_cloud(lidar,
                                      side_range=(-15,15),
                                      fwd_range=(-20,20),
                                      res=0.05,
                                      min_height = 0,
                                      max_height = 1,
                                      saveto=None)





    lidar_db_repr = birds_eye_point_cloud(lidar_mean_filter_db,
                                      side_range=(-15,15),
                                      fwd_range=(-20,20),
                                      res=0.05,
                                      min_height = 0,
                                      max_height = 1,
                                      saveto=None)


    lidar_notaug= birds_eye_point_cloud(dataveloaugrot,
                                      side_range=(-15,15),
                                      fwd_range=(-20,20),
                                      res=0.05,
                                      min_height = 0,
                                      max_height = 1,
                                      saveto=None)
    
    background = np.dstack([lidar_repr,lidar_repr,lidar_repr])

    points_marks = np.nonzero(lidar_db_repr)
    
    plt.figure(str(ii))

    plt.imshow(background, cmap='gray')

    plt.scatter(points_marks[1],points_marks[0],color='y',s=5)
    plt.axis('off')
    #plt.show()
    plt.savefig('video'+str(ii)+'.png')

#%%





#%%

#DBSCAN part


#0.06 i 5 valors 
#%%




"""
cluster_df = lidar_mean_filter_db

lines = []

for cluster in range(n_clusters_):
    sub_cluster_df = cluster_df[cluster_df[:,4] == cluster]
    points = sub_cluster_df[:,0:3].values
    distances = squareform(pdist(points))
    for i in range(0,15):
        max_index = np.argmax(distances)
        i1, i2 = np.unravel_index(max_index, distances.shape)
        distances[i1,i2] = 0.0
    max_dist = np.max(distances)
    max_index = np.argmax(distances)
    i1, i2 = np.unravel_index(max_index, distances.shape)
    p1 = sub_cluster_df.iloc[i1]
    p2 = sub_cluster_df.iloc[i2]
    lines.append(([p1["Easting"], p2["Easting"]],[p1["Northing"], p2["Northing"]], [p1["Altitude"], p2["Altitude"]]))

"""




#%%









#%%

