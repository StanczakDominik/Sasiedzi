import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial
import time
N=1000000
S=10

np.random.seed(1)
r=np.random.random((N,2))

def naive_distances(r, particle_index):
    start_time=time.time()
    r_particle = r[particle_index,:]

    distances = np.sqrt(np.sum((r-r_particle)**2, axis=1))
    sorted_arguments = np.argsort(distances)[1:S+1]
    runtime = time.time() - start_time
    return sorted_arguments, runtime

def create_tree(r):
    return scipy.spatial.cKDTree(r)

def tree_distances(r, particle_index, how_many_attempted):
    start_time=time.time()
    tree = scipy.spatial.cKDTree(r)
    r_particle = r[particle_index,:]
    distances_close, r_close_indices = tree.query(r_particle, k=how_many_attempted)
    r_close = r[r_close_indices]
    distances = np.sqrt(np.sum((r_close-r_particle)**2, axis=1))
    sorted_arguments = r_close_indices[np.argsort(distances)[1:S+1]]
    runtime = time.time() - start_time
    return sorted_arguments, runtime

sorted_arguments, naive_time = naive_distances(r, 0)
r_closest = r[sorted_arguments, :]
x, y = r[:,0], r[:,1]
x_closest, y_closest = r_closest[:,0], r_closest[:,1]

tree_indices, tree_time = tree_distances(r, 0, 100)

print(sorted_arguments-tree_indices)
print (naive_time, tree_time, naive_time-tree_time)

plt.plot(x, y, 'bo', alpha=0.5)
plt.plot(x_closest, y_closest, 'go')
plt.plot(x[0], y[0], 'ro', alpha=0.9)
plt.grid()
plt.xlabel("x")
plt.ylabel("y")
plt.xlim(0,1)
plt.ylim(0,1)
plt.show()
