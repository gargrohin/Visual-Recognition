import numpy as np
import os
import cv2

from joblib import dump,load

from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.pairwise import euclidean_distances
import math

clust = load("clusters.joblib")

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
sift = cv2.xfeatures2d.SIFT_create()
for k in range(len(images)):
    for j in range(count[d]):
        word = clust.labels_[i]
        '''if kp[d][j].pt[0] > 50 and kp[d][j].pt[0] < 280:
            if kp[d][j].pt[1] > 260 and kp[d][j].pt[1] < 370:
                nid[d][word] += 100
            else:
                nid[d][word] += 1
        else:
            nid[d][word] += 1'''
        if indoc[word]==0:
            indoc[word]=1
            ni[word]+=1
            nd[d]+=1
        nid[d][word]+=1
        i+=1
    d+=1
    indoc = np.zeros(words)

weights = np.zeros([N,words])

for d in range(N):
    for i in range(words):
        weights[d][i] = (nid[d][i]/nd[d]) * math.log10(N/ni[i])

np.savez('tfidf.npz',weights)