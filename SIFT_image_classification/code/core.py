import numpy as np
import os
import cv2

from joblib import dump,load

from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
#from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
import math

def get_sift(path):

    sift = cv2.xfeatures2d.SIFT_create()

    #desc ={}
    #keys = {}
    alldes = np.empty([1,128])
    for objdir in os.listdir(path):
        obj = path + objdir
        #kp=[]
        #des=[]
        for img in os.listdir(obj):
            img1 = obj + '/' + img
            img1 = cv2.imread(img1)
            img1= img1[:300,250:400,:]
            kp1,des1 = sift.detectAndCompute(img1,None)
            #kp.append(kp)
            #des.append(des1)
            alldes = np.concatenate([alldes,des1])
            print('{}'.format(img),end="\r")
        #keys[objdir] = kp
        #desc[objdir] = des
        print('{} done'.format(objdir))
    alldesc_real = alldes[1:,:]
    return alldesc_real


path = os.getcwd()
path = path[:path.rfind('/')] + '/train/'

#alldesc = get_sift(path)

#np.savez("sift_freatures.npz",alldesc)

alldesc = np.load("sift_freatures.npz")['arr_0']


clust = MiniBatchKMeans(n_clusters=12500,max_no_improvement=None,batch_size=10000).fit(alldesc)

dump(clust,"clusters_2000.joblib")


"""

words = np.shape(clust.cluster_centers_)[0]
N = 3456
w = np.zeros(words)
idf = np.zeros([N,words])

indoc = np.zeros(words)
nd = np.zeros(N)
ni = np.zeros(words)
nid = np.zeros([N,words])
i=0
d=0

for obj in os.listdir(path):
    for img in desc[obj]:
        for des in range(np.shape(img)[0]):
            word = clust.labels_[i]
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


test_path = path[:path.rfind('/')]
test_path = test_path[:test_path.rfind('/')] + '/sample_test/'

sift = cv2.xfeatures2d.SIFT_create()
for img in os.listdir(test_path):
    if img == 'instances.txt':
        continue
    imr = test_path + img
    test = cv2.imread(imr)
    kpt, dest = sift.detectAndCompute(test,None)
    query=np.zeros(words)
    j = clust.predict(dest)
    for x in j:
        query[x]+=1
    break


#sift = cv2.xfeatures2d.SIFT_create()
img = 'easy_single_3.jpg'
imr = test_path + img
test = cv2.imread(imr)
kpt, dest = sift.detectAndCompute(test,None)
query=np.zeros(words)
j = clust.predict(dest)
for x in j:
    query[x]+=1


x = euclidean_distances(np.array([query]), weights)

y = x[0]
y = np.argsort(y)

images = []
for objdir in os.listdir(path):
    obj = path + objdir
    for img in os.listdir(obj):
        img_path = obj + '/' + img
        images.append(img_path[32:])


for i in range(N):
    print(images[y[i]])

    """