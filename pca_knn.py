import numpy as np
import pickle
import sys

from sklearn.decomposition import PCA

# From https://www.cs.toronto.edu/~kriz/cifar.html
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def to_grayscale(data):
	""" Converts CIFAR 10 pixel data in RGB format to grayscale.
	L(Gray) = 0.299R + 0.587G + 0.144B

	Parameters:
		data (list of lists): dimensionality is the number of data points by 3072 (32^2 * 3)
	Returns:
		List with dimension of the number of data points
	"""
	return_value = []
	for data_point in data:
		grayed_data_point = []
		for R, G, B in zip(data_point[:1024], data_point[1024:2048], data_point[2048:]):
			grayed_data_point.append(0.299*R + 0.587*G + 0.114*B)
		return_value.append(grayed_data_point)
	return return_value

def eu_distance(x, y):
	""" Calculates the Euclidean distance between two points of the same dimension

	Parameters:
		X (list): A multidimensional point
		Y (list): A multidimensional point
	"""
	total_distance = 0
	for xdim, ydim in zip(x, y):
		total_distance += (xdim-ydim)**2
	return np.sqrt(total_distance)

def nearest_neighbors(training_data, training_data_labels, test_sample, K):
	""" K-Nearest Neighbors search for the nearest neighbors in the 
	training data.
	"""
	# Don't need to exclude test sample from training data
	dists = [eu_distance(data, test_sample) for data in training_data]
	dist_arr = np.array(dists)
	knn_indices = dist_arr.argsort()[:K] # Indices of nearest neighbors in training data and labels
	voted_label = {}
	for k_index in knn_indices:
		voting_weight = 1.0/eu_distance(training_data[k_index], test_sample) # Inverse Euclidean distance
		k_label = training_data_labels[k_index]
		if k_label in voted_label:
			voted_label[k_label] += voting_weight
		else:
			voted_label[k_label] = voting_weight
	# Finding the key for the max value in labels dictionary
	maxVal = 0
	voted_key = None
	for key, val in voted_label.items():
		if val > maxVal:
			voted_key = key
			maxVal = val
	return voted_key
	# from operator import itemgetter
	# return max(voted_label.items(), key=itemgetter(1))[0]


def main(K, D, N, data_path):
	data_dict = unpickle(data_path)
	# str.encode('UTF-8') and str.decode('UTF-8')
	labels = data_dict['labels'.encode('UTF-8')]
	data = data_dict['data'.encode('UTF-8')]

	sub_labels = labels[:1000]
	sub_batch = data[:1000]

	grayed_data = to_grayscale(sub_batch)

	# test_set = sub_batch[:N]
	# training_set = sub_batch[N:]
	test_set_labels = sub_labels[:N]
	test_set = grayed_data[:N]

	training_set_labels = sub_labels[N:]
	training_set = grayed_data[N:]


	# import pdb; pdb.set_trace()
	# PCA
	pca = PCA(n_components=D, svd_solver='full')
	pca.fit(training_set)

	reduced_training_data = pca.transform(training_set)
	reduced_test_data = pca.transform(test_set)
	# At this point, each data point should have been reduced from 3072 -> 1024 -> D
	# K-NN
	classifications = [nearest_neighbors(reduced_training_data, training_set_labels, sample, K) for sample in reduced_test_data]
	with open("out_file.txt", 'w') as out_file:
		for index in range(len(test_set_labels)):
			out_file.write("{} {}\n".format(classifications[index], test_set_labels[index]))



if __name__ == '__main__':
	K, D, N, data_path = sys.argv[1:5]
	main(int(K), int(D), int(N), data_path)