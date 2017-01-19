#!/bin/python
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import matrix_rank
from numpy.linalg import norm
from scipy.spatial.distance import cosine
import math

def pca(m, k):
    from numpy.linalg import svd
    from numpy.linalg import eig
    from numpy.linalg import det
    u,s,v = svd(m)
    rs = np.sqrt(np.diag(s[:k]))
    x=np.dot(u[:,:k], rs)
    y=np.dot(rs, v[:k])
    mhat=np.dot(x, y)
    return s, x, y, mhat

def plot(m):
	plt.figure()
	img=plt.imshow(m)
    #img.set_clim(0.0,1.0)
	img.set_interpolation('nearest')
    #plt.set_cmap('gray')
	plt.colorbar()

def term_doc_matrix():
	N = 12
	D = 9
	m = np.zeros((N,D))
	# Documents taken from http://lsa.colorado.edu/papers/dp1.LSAintro.pdf
	docs = [
		[ [0,1], [1,1], [2,1] ],
		[ [2,1], [3,1], [4,1], [5,1], [6,1], [8,1] ],
		[ [1,1], [3,1], [4,1], [7,1] ],
		[ [0,1], [4,2], [7,1] ],
		[ [3,1], [5,1], [6,1] ],
		[ [9,1] ],
		[ [9,1], [10,1] ],
		[ [9,1], [10,1], [11,1] ],
		[ [8,1], [10,1], [11,1] ],
	]
	# fill matrix
	for i in xrange(len(docs)):
		d = docs[i]
		for w,tf in d:
			m[w][i] = tf
	return m

def clustering(m, k):
	from sklearn.cluster import KMeans
	c = np.zeros((m.shape[1],k))
	y_pred = KMeans(n_clusters=k).fit_predict(m.T)
	for i in xrange(len(y_pred)):
		c[i][y_pred[i]] = 1
	return c

def all_col_dist(m):
	D = m.shape[1]
	d = np.zeros((D,D))
	for i in xrange(D):
		div = m[:,i]
		for j in xrange(D):
			djv = m[:,j]
			d[j][i] = cosine(div,djv)
	return d

if __name__ == "__main__":
	m = term_doc_matrix()
	plot(m)
	plt.savefig("lsa-tfm.png")
	d = all_col_dist(m)
	plot(d)
	plt.savefig("lsa-dists.png")
	k = 2
	c = clustering(m, 2)
	plot(c)
	plt.savefig("lsa-clusters.png")
	s,wv,dv,mhat = pca(m,k)
	plot(wv)
	plt.savefig("lsa-wordv.png")
	plot(dv)
	plt.savefig("lsa-docv.png")
	plt.figure()
	plt.plot(dv[0], dv[1], 'bo')
	plt.savefig("lsa-docv-plot.png")
	plot(mhat)
	plt.savefig("lsa-recon-tfm.png")
	d = all_col_dist(mhat)
	plot(d)
	plt.savefig("lsa-recon-dists.png")
