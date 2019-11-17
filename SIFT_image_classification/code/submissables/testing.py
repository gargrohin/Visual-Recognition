import numpy as np
import os
import cv2

from joblib import dump,load

#from sklearn.cluster import KMeans
#from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.pairwise import euclidean_distances
import math

from scipy.spatial.distance import cosine

weights = np.load('tfidf.npz')['arr_0']   # load the tf-idf weights
images = np.load('images_full.npz')['arr_0']   # load the train-images order.
clust = load('clusters_125000.joblib')    # load the K-Means Cluster

words = len(clust.cluster_centers_)

N = 3456  # total train documents


### Set path to test images###

test_path = os.getcwd() 
test_images = "/test" 
test_path = test_path + test_images    # manually enter the folder's path where all the test images are




max_dist = 300.0 # threshold

sift = cv2.xfeatures2d.SIFT_create(sigma = 1.5)  

for img in os.listdir(test_path):
	if img=="instances.txt":
		continue
	img_path = test_path +'/' + img
	test = cv2.imread(img_path)
	_, dest = sift.detectAndCompute(test,None)

	query=np.zeros(words)  # our query vector

    ## removing descriptors outside the threshhold of clusters
	for x in dest:
		dist = euclidean_distances(clust.cluster_centers_[clust.predict(np.array([x]))],np.array([x]))[0][0]
		if dist < max_dist:
			query[clust.predict(np.array([x]))] +=1
    
    # similarity vector
	sim=[]
	for i in range(N):
 	   l = cosine(np.array([query]),weights[i])  # cosine_similarity is used
 	   sim.append(l)

	y = np.argsort(sim)  ## rank the index of train images according to similarity vector
 	
	with open("results_final/"+ img[:img.rfind('.j')]+".txt",'w') as f:
 		for i in range(N):
 			out = images[y[i]][32:images[y[i]].rfind('/')] + '_' + images[y[i]][images[y[i]].rfind('/')+1:]
 			print(out,file=f)
	print("{} done".format(img))

	## al output files in folder: results