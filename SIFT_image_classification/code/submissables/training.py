import numpy as np
import os
import cv2

from joblib import dump,load

from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.pairwise import euclidean_distances
import math

def get_sift(path):  # function to extract sift features

    sift = cv2.xfeatures2d.SIFT_create()

    desc ={}
    images=[]
    c=[]
    keys = []
    alldes = np.empty([1,128])
    for objdir in os.listdir(path):
        obj = path + objdir
        des=[]
        for img in os.listdir(obj):
            img1 = obj + '/' + img
            images.append(img1)
            img1 = cv2.imread(img1)
            kp1,des1 = sift.detectAndCompute(img1,None)
            keys.append(kp1)
            des.append(des1)
            c.append(des1.shape[0])
            alldes = np.concatenate([alldes,des1])
            print('{}'.format(img),end="\r")
        desc[objdir] = des
        print('{} done'.format(objdir))
    alldesc_real = alldes[1:,:]
    return keys,c,images,desc,alldesc_real


path = os.getcwd()
path = path[:path.rfind('/')] + '/train/'

kp,count,images,des, alldesc = get_sift(path)

np.savez('images_full.npz',images)

#np.savez("sift_freatures.npz",alldesc)

#alldesc = np.load("sift_freatures.npz")['arr_0']

## K-Means clustering
clust = MiniBatchKMeans(n_clusters=12500,max_no_improvement=None,batch_size=5000).fit(alldesc)

dump(clust,"clusters_12500.joblib")

clust = load("clusters_12500.joblib")

words = np.shape(clust.cluster_centers_)[0]
N = 3456
w = np.zeros(words)
idf = np.zeros([N,words])

### TF-IDF calculations ###

indoc = np.zeros(words)
nd = np.zeros(N)
ni = np.zeros(words)
nid = np.zeros([N,words])
i=0
d=0


for k in range(len(images)):
    for j in range(count[d]):
        word = clust.labels_[i]
        if kp[d][j].pt[1] > 51 and kp[d][j].pt[1] < 281:                   # Give more weight to center of the images
            if kp[d][j].pt[0] > 261 and kp[d][j].pt[0] < 372:              # where the object is present
                nid[d][word] += 200
            else:
                nid[d][word] += 1
        else:
            nid[d][word] += 1
        if indoc[word]==0:
            indoc[word]=1
            ni[word]+=1
        i+=1
    d+=1
    indoc = np.zeros(words)

for d in range(N):
    nd[d]= sum(nid[d])
for d in range(N):
    nid[d] = nid[d]/nd[d]    # normalization

## weight matrix for each image##
weights = np.zeros([N,words])

for d in range(N):
    for i in range(words):
        weights[d][i] = (nid[d][i]) * math.log10(N/ni[i])

np.savez('tfidf.npz',weights)
