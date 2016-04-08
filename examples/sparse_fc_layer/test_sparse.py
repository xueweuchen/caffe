import caffe
import numpy as np
from scipy.sparse import csr_matrix

indptr = np.array([0, 2, 3, 6])
indices = np.array([0, 2, 2, 0, 1, 2])
data = np.array([1, 2, 3, 4, 5, 6])
csr = csr_matrix((data, indices, indptr), shape=(3, 10)).toarray()

net = caffe.Net("sparse_net.prototxt", caffe.TRAIN)
net.blobs['value'].data[...] = data
net.blobs['indices'].data[...] = indices
net.blobs['ptr'].data[...] = indptr

weight = net.params['spfc'][0].data.T
y = csr.dot(weight)
y_net = net.forward()['spfc']
diff = y - y_net
assert (np.abs(diff) < 0.001).all()
