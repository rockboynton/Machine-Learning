import unittest

import numpy as np
from scipy import spatial

from sklearn.datasets import make_blobs

from kmeans import KMeans

EPS = 1e-1
N_TRIALS = 100
N_ITER = 20

def find_smallest_distances(X, V):
    """
    Finds the closest match (smallest distance) between
    the rows of the matrix V and the rows in the matrix X.  Returns
    the smallest distance found for every row in V.
    """
    
    # returns a matrix of X.shape[0] rows and V.shape[0] columns
    distances = spatial.distance.cdist(X, V, "euclidean")
    
    # find index of the minimum distance for each entry in V
    # returns V.shape[0]-length vector of indices
    closest_idx = np.argmin(distances, axis=0)
    
    # get the closest distance for each point
    closest_dist = np.diag(distances[closest_idx, :])
    
    return closest_dist

def generate_cluster_samples():
    """
    Generates random samples in a 2D space
    from 4 clusters.
    
    Returns tuple of feature matrix, labels, and center coordinates.
    """
    n_samples = 100
    var = 0.001
    
    # 4 clusters in a 2D space
    centers = np.array([[0, 0],
                        [0, 1],
                        [1, 0],
                        [1, 1]])
   
    X, y = make_blobs(n_samples,
                      centers=centers,
                      cluster_std = np.sqrt(var),
                      shuffle=False)
    
    return X, y, centers

class TestKMeans(unittest.TestCase):
    def test_initialize(self):
        """
        Tests initialize methods of the KMeans class. 
        """
        k = 3
        n_samples = 100
        n_features = 10
        
        for i in range(N_TRIALS):
            X = np.random.randn(n_samples, n_features)
        
            kmeans = KMeans(k, N_ITER)
            kmeans.initialize_clusters(X)
        
            # ensure that the cluster_centers matrix has the right shape
            self.assertEqual(kmeans.cluster_centers.ndim, 2)
            self.assertEqual(kmeans.cluster_centers.shape[0], k)
            self.assertEqual(kmeans.cluster_centers.shape[1], n_features)
        
            # Check that every center is one the points in X.
            # Calculcate the distances between every cluster center
            # and every point in X.  Find the closest matches.
            # Checks that the distances are nearly 0.0
            distances = find_smallest_distances(X, kmeans.cluster_centers)
            for d in distances:
                self.assertAlmostEqual(d, 0.0)
                
    def test_assign_points(self):
        """
        Tests initialize methods of the KMeans class. 
        """
        X, y, centers = generate_cluster_samples()
        n_samples = X.shape[0]
        k = centers.shape[0]
    
        kmeans = KMeans(k, N_ITER)
        
        # Set cluster centers so that assignment is deterministic
        kmeans.cluster_centers = centers
        assignments, distances = kmeans.assign_points(X)

        # check assignment array shape
        self.assertEqual(assignments.ndim, 1)
        self.assertEqual(assignments.shape[0], n_samples)
        
        # check distances array shape
        self.assertEqual(distances.ndim, 1)
        self.assertEqual(distances.shape[0], n_samples)
        
        # check that assignments only include valid cluster indices (0 <= idx < k)
        self.assertTrue(np.all(np.logical_and(assignments < k, assignments >= 0)))
        
        # Check cluster assignments are correct
        self.assertTrue(np.all(assignments[:25] == 0))
        self.assertTrue(np.all(assignments[25:50] == 1))
        self.assertTrue(np.all(assignments[50:75] == 2))
        self.assertTrue(np.all(assignments[75:] == 3))
        
    def test_reinitialize_empty_clusters(self):
        """
        Tests reassignment of points to empty clusters
        """
        X, y, centers = generate_cluster_samples()
        n_samples = X.shape[0]
        k = centers.shape[0]
    
        kmeans = KMeans(k, N_ITER)
        
        # Set cluster centers so that assignment is deterministic
        kmeans.cluster_centers = centers
        assignments, distances = kmeans.assign_points(X)
        
        # reassign all points in cluster 3 to cluster 2 to create empty cluster
        assignments[75:] = 2
        
        # reinitialize empty clusters by reassigning points
        assignments = kmeans.reinitialize_empty_clusters(X, assignments, distances)
        
        # ensure that each cluster has an assigned point
        # and that only valid cluster indices are used
        self.assertSetEqual(set(assignments), set(range(k)))
        
    def test_update_centers(self):
        """
        Tests update centers
        """
        X, y, centers = generate_cluster_samples()
        n_samples = X.shape[0]
        n_features = X.shape[1]
        k = centers.shape[0]
    
        kmeans = KMeans(k, N_ITER)
        
        # Set cluster centers so that assignment is deterministic
        kmeans.cluster_centers = centers
        assignments, distances = kmeans.assign_points(X)
        assignments = kmeans.reinitialize_empty_clusters(X, assignments, distances)
        
        # clear out centers to test method
        kmeans.cluster_centers = np.zeros((k, n_features))
        kmeans.update_centers(X, assignments)
        
        # calculate average difference in coordinates of estimated
        # and real centers
        error = np.linalg.norm(kmeans.cluster_centers - centers) / k
        self.assertLess(error, EPS)
        
    def test_score(self):
        """
        Tests within-cluster variance
        """

        X, y, centers = generate_cluster_samples()
        n_samples = X.shape[0]
        n_features = X.shape[1]
        k = centers.shape[0]
    
        kmeans = KMeans(k, N_ITER)
        assignments = kmeans.fit_predict(X)

        score = np.sqrt(kmeans.score(X)) / n_samples
        self.assertLess(score, EPS)
        
    def test_whole(self):
        """
        Tests the score method.
        """
        
        X, y, centers = generate_cluster_samples()
        n_samples = X.shape[0]
        n_features = X.shape[1]
        k = centers.shape[0]
        
        # run N_TRIALS, pick best model
        best_model = None
        for i in range(N_TRIALS):
            kmeans = KMeans(k, N_ITER)
            kmeans.fit(X)
            if best_model is None:
                best_model = kmeans
            elif kmeans.score(X) < best_model.score(X):
                best_model = kmeans
        
       
        # check sum squared errors
        sum_squared_errors = best_model.score(X)
        self.assertLess(sum_squared_errors / n_samples, EPS)
        
        # compare centers to expected centers
        smallest_distances = find_smallest_distances(best_model.cluster_centers, centers)
        for distance in smallest_distances:
            self.assertLess(distance, EPS)
        
        
if __name__ == "__main__":
    unittest.main()