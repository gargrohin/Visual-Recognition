## Create a folder containing a list of all training images
import os
train_path = '/Users/apple/tensorflow3/QBH/MTG-QBH/assignment_train'
list_of_folders=[]
for folder in os.listdir(train_path) :
    if(folder!='.DS_Store') : list_of_folders.append(train_path+'/'+folder)
list_of_files=[]
for folder in list_of_folders :
    for file in os.listdir(folder) :
        if(file!='.DS_Store') : list_of_files.append(folder+'/'+file)

## Define a function to extract all the features from all of the images
import cv2
sift = cv2.xfeatures2d.SIFT_create(nfeatures=500)
import numpy as np

def create_features(list_of_files) :
    features = []
    for file in list_of_files :
        keypoint, descriptor = sift.detectAndCompute(cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2GRAY), None)
        for d in descriptor : features.append(d)
        del keypoint, descriptor
    features = np.array(features)
    return features

## Define a function to construct the vocabulary tree
from sklearn.cluster import MiniBatchKMeans
number_of_branches = 5
leaf_clusters_size = 20
max_depth = 5
cluster_count = 0
tree = {}
clusters = {}
Model = MiniBatchKMeans(n_clusters = number_of_branches)
inverted_list = {}

def construct_tree(cluster_index, feature_index, depth) :
    global tree, clusters, max_depth, features, cluster_count
    tree[cluster_index] = []
    if len(feature_index) >= leaf_clusters_size and depth < max_depth :
        Model.fit([features[i] for i in feature_index])
        child_feature_index = [[] for i in range(number_of_branches)]
        for i in range(len(feature_index)) : child_feature_index[Model.labels_[i]].append(feature_index[i])
        for i in range(number_of_branches) :
            cluster_count += 1
            clusters[cluster_count] = Model.cluster_centers_[i]
            tree[cluster_index].append(cluster_count)
            construct_tree(cluster_count, child_feature_index[i], depth+1)
    else : inverted_list[cluster_index] = {}

## Define a function to find the leaf cluster closest to a given SIFT descriptor
def leaf_cluster_index(descriptor, cluster_index) :
    global tree, clusters
    min_distance = float('inf')
    descendant = None
    for child in tree[cluster_index] :
        distance =  np.linalg.norm([clusters[child]-descriptor])
        if min_distance > distance :
            min_distance = distance
            descendant = child
    if tree[descendant] == [] : return descendant
    else : return leaf_cluster_index(descriptor, descendant)

## Define a function to create the inverted list for an image
def create_inverted_list(list_of_files) :
    global inverted_list
    for filepath in list_of_files :
        keypoint, descriptor = sift.detectAndCompute(cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2GRAY), None)
        for d in descriptor :
            closest_leaf_index = leaf_cluster_index(d,0)
            if filepath in inverted_list[closest_leaf_index] : inverted_list[closest_leaf_index][filepath] += 1
            else : inverted_list[closest_leaf_index][filepath] = 1
        del keypoint, descriptor


## Define a function to find the matching scores with the database images for a given query
import operator
def score_of_images(query_filepath) :
    global inverted_list
    keypoint, descriptor = sift.detectAndCompute(cv2.cvtColor(cv2.imread(query_filepath), cv2.COLOR_BGR2GRAY), None)
    scores = {}
    for d in descriptor:
        closest_leaf_index = leaf_cluster_index(d,0)
        for filepath in inverted_list[closest_leaf_index] :
            if filepath in scores : scores[filepath] += 1
            else : scores[filepath] = 1
    del keypoint, descriptor
    list_of_scores = sorted(scores.items(), key=operator.itemgetter(1))
    list_of_scores.reverse()
    return scores, list_of_scores

## Main - create the vocabulary tree and the inverted list

features = create_features(list_of_files)
feature_index = [k for k in range(len(features))]
construct_tree(0,feature_index,0)
del features
create_inverted_list(list_of_files)

# Load the test images and find the score vector for the database images
test_path = '/Users/apple/tensorflow3/QBH/MTG-QBH/assignment_test'
list_of_images=[]
for image in os.listdir(test_path) :
    if(image!='.DS_Store') : list_of_images.append(test_path+'/'+image)

Scores = [{} for i in range(len(list_of_images))]
List_of_Scores = [[] for i in range(len(list_of_images))]
for i in range(len(list_of_images)) :
    Scores[i], List_of_Scores[i] = score_of_images(list_of_images[i])


