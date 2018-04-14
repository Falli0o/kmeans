
# coding: utf-8

# In[1]:

#Use python 2.7
from __future__ import division
#get_ipython().magic(u'matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D


# In[2]:


murder = np.loadtxt('murderdata2d.txt')


# In[3]:


dataTrain = np.loadtxt('IDSWeedCropTrain.csv', delimiter = ',')
dataTest = np.loadtxt('IDSWeedCropTest.csv', delimiter = ',')


# In[4]:


XTrain = dataTrain[:, : -1]
YTrain = dataTrain[:, -1]
XTest = dataTest[:, : -1]
YTest = dataTest[:, -1]


# In[5]:


def pca(data):
    cov_matrix = np.cov(data.T)
    eigenvalue, eigenvector = np.linalg.eig(cov_matrix) #the normalized_eigenvector is the unit vector
    #sort the eigenvectors by the eigenvalue (from large to small)
    index = np.argsort(eigenvalue)[::-1]
    sorted_unit_vector_matrix = eigenvector.T[(index)].T
    sorted_eigenvalues = eigenvalue[(index)]
    return (sorted_eigenvalues, sorted_unit_vector_matrix) #the sorted unit vector is column wise


# In[6]:


def transform_data(data, d):
    n_component = data.shape[1]
    if d <= n_component:
        #mean_matrix = np.tile(np.mean(data, axis = 0), data.shape[0]).reshape(data.shape[0], -1)
        #mean_0_data_matrix = data - mean_matrix
        #return np.dot(pca(data)[1].T, mean_0_data_matrix.T).T[:, :d]
        return np.dot(pca(data)[1].T, data.T).T[:, :d]
    else:
        print 'the dimensions should not be larger than the numbers of original components'


# In[7]:


#transformed_murder_result = transform_data(murder, 2)
variance = pca(murder)[0]
print variance
murder_principal_vector = pca(murder)[1]
print murder_principal_vector
std_1 = math.sqrt(variance[0])
std_2 = math.sqrt(variance[1])
scaled_unit_vector_1 = np.array(murder_principal_vector[:, 0])*std_1
scaled_unit_vector_2 = np.array(murder_principal_vector[:, 1])*std_2
print scaled_unit_vector_1, scaled_unit_vector_2


# In[8]:


mean_murder_matrix = np.tile(np.mean(murder, axis = 0), murder.shape[0]).reshape(murder.shape[0], -1)
mean_0_murder_matrix = murder - mean_murder_matrix
fig1 = plt.figure(1)
fig1.set_size_inches(10, 8)
plt.axis('equal')

plt.scatter(mean_0_murder_matrix[:,0], mean_0_murder_matrix[:, 1]);
plt.scatter(0,0)

plt.arrow(0, 0, scaled_unit_vector_1[0], scaled_unit_vector_1[1],                head_width=0.3, head_length=1, fc='k', ec='k');
plt.arrow(0, 0, scaled_unit_vector_2[0], scaled_unit_vector_2[1],                head_width=0.3, head_length=1, fc='k', ec='k');
plt.title("Normalized data: \n Principal eigenvectors pointing out of the mean");
plt.savefig("Normalized data");


# In[9]:


fig2 = plt.figure(2)
fig2.set_size_inches(10, 8)
plt.axis('equal')

plt.scatter(murder[:,0], murder[:, 1]);
mean_point = np.mean(murder, axis = 0)

plt.scatter(mean_point[0],mean_point[1])

plt.arrow(mean_point[0],mean_point[1], scaled_unit_vector_1[0], scaled_unit_vector_1[1],                head_width=0.3, head_length=1, fc='k', ec='k');
plt.arrow(mean_point[0],mean_point[1], scaled_unit_vector_2[0], scaled_unit_vector_2[1],                head_width=0.3, head_length=1, fc='k', ec='k');
plt.title("Non-normalized data: \n Principal eigenvectors pointing out of the mean");
plt.savefig("Non-normalized data");


# In[10]:


fig3 = plt.figure(3)
var_xtrain = pca(XTrain)[0]
plt.plot(range(len(var_xtrain)), var_xtrain); 
plt.scatter(range(len(var_xtrain)), var_xtrain, c = 'orange');
plt.ylabel("variance");
plt.xlabel("principal components index");
plt.title("Variance versus principal components index");
plt.savefig("Variance versus principal components index");


# In[11]:


cumul_var = np.cumsum(var_xtrain/np.sum(var_xtrain))
print cumul_var
fig4 = plt.figure(4)
plt.plot(cumul_var)
plt.scatter(range(len(cumul_var)), cumul_var, c = 'orange');
plt.axhline(0.9);
plt.axhline(0.95);
plt.ylabel("cumulative normalized variance");
plt.xlabel("principal components index");
plt.title("Cumulative normalized variance versus principal components index");
plt.savefig("Cumulative normalized variance versus principal components index");


# In[12]:


def mds(data, d):
    from mpl_toolkits.mplot3d import Axes3D
    transformed_data = transform_data(data, d)
    if d == 2:
        fig = plt.figure()
        plt.scatter(transformed_data[:, 0], transformed_data[:, 1], alpha = 0.5)
    elif d == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        xs = transformed_data[:, 0]
        ys = transformed_data[:, 1]
        zs = transformed_data[:, 2]
        ax.scatter(xs, ys, zs)
    else:
        print "only 2 or 3 dimensions can be plotted"
    


# In[13]:


mds(XTrain, 4)


# In[14]:


mds(XTrain, 3)
plt.title("Pesticide 3d");
plt.savefig("pesticide 3d");


# In[15]:


mds(XTrain, 2)
plt.title("Pesticide 2d");
plt.savefig("pesticide 2d");


# In[16]:


def first_k_initial_point(data, k):
    start_point = {}
    for i in range(k):
        start_point[i] = data[i]
    return start_point.values() 


# In[17]:


def random_initial_point(data, k):
    start_point = {}
    import numpy as np
    random_pool = data.shape[0]
    index = np.random.choice(random_pool, size = k, replace = False)
    for i in range(k):
        start_point[i] = data[index[i]]
    return start_point.values()


# In[18]:


def kmeans(data, k, initial_point):
    
    if initial_point == 'first_k_point':
        start_point = first_k_initial_point(data, k)
        centroid = start_point
    elif initial_point == 'random_point':
        start_point = random_initial_point(data, k)
        centroid = start_point
        
    iteration_times = 1
    
    while True:
        subset = {}
        dis = []
        for centroid_point in centroid:
            centroid_matrix = np.tile(centroid_point, data.shape[0])
            centroid_matrix = centroid_matrix.reshape(-1,data.shape[1])
            for_cal_dis = np.subtract(centroid_matrix, data)
            dis_matrix = np.dot(for_cal_dis, for_cal_dis.T)
            #try to find the diagonal
            distance = np.diag(dis_matrix)
            dis.append(distance)
        cluster_index = np.argsort(np.concatenate(dis).reshape(k, -1), axis = 0)[0]

    #index = np.where(cluster_index == 1)
        for i in range(k):
            subset[i] = data[np.where(cluster_index == i)]
        re_centroid = []
        for cluster in subset.values():
            re_centroid.append(np.mean(cluster, axis = 0))
        diff = np.array(re_centroid[0]) - np.array(centroid[0])
        dis = np.dot(diff, diff.T)
        #print dis
        if dis == 0: #and re_centroid[1] == centroid[1]: #change it because k can be larger than 2
            print "iteration_times: " + str(iteration_times)
            break
        else:
            centroid = re_centroid
            iteration_times += 1
    return re_centroid


# In[19]:


kmeans(XTrain, 2, initial_point = 'first_k_point')


# In[20]:


centroids = kmeans(XTrain, 2, initial_point = 'first_k_point')


# In[21]:


c1 = []
c2 = []
for j in centroids[0]:
    c1.append(np.around(np.array(j), decimals = 4))
for j in centroids[1]:
    c2.append(np.around(np.array(j), decimals = 4))
print c1
print c2


# In[22]:


kmeans(XTrain, 2, initial_point = 'random_point')


# In[23]:


from sklearn.cluster import KMeans
startingPoint = np.vstack((XTrain[0,],XTrain[1,]))
sk_kmeans = KMeans(n_clusters=2, n_init=1, init=startingPoint).fit(XTrain)
sk_kmeans.cluster_centers_


# In[24]:


experiments = {}
for i in range(100):
    experiments[i] =kmeans(XTrain, 2, initial_point = 'random_point')
experiments.values()    

