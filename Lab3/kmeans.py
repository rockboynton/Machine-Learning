import random

import numpy as np
from scipy import spatial

class KMeans:
    def __init__(self, k, max_iter):
        self.k = k
        self.max_iter = max_iter
        
        # this will need to be created by the appropriate method
        self.cluster_centers = None
        
    def initialize_clusters(self, X):
        """
        Choose initial cluster centers
        """
        self.cluster_centers = X[np.random.choice(np.arange(len(X)), self.k), :]

        
    def assign_points(self, X):
        """
        Finds the closest match (smallest distance) between
        the rows of the matrix X and the rows in the matrix
        cluster_centers.  Returns a tuple of cluster assignments
        and minimum distances.  The cluster assignments are stored
        as a 1D array of the index (0 to k - 1) of the closest 
        cluster center for each point. The minimum distances are
        stored as a 1D array of the distances between every point
        and its assigned cluster center.
        """
        distances = spatial.distance.cdist(X, self.cluster_centers, metric='euclidean')
        cluster_assignments = np.argmin(distances, axis=1)
        min_distances = np.min(distances, axis=1)
        return cluster_assignments, min_distances

        
    def reinitialize_empty_clusters(self, X, cluster_assignments, min_distances):
        """
        If any clusters are empty (no points assigned), assign points farthest
        away from their clusters' centers to the empty clusters.  Updates
        cluster assignments and minimum distances as appropriate.
        """
        # ? should i sort every loop or only sort at the start and use an index
        for cluster in range(self.k):
            if cluster not in np.unique(cluster_assignments):
                new_assignment = np.argsort(min_distances)[-1]
                cluster_assignments[new_assignment] = cluster
                min_distances[new_assignment] = 0

        return cluster_assignments

        
    def update_centers(self, X, cluster_assignments):
        """
        Re-calculate cluster centers using points assigned to each cluster.
        """
        for cluster in range(self.k):
            cluster_points = X[cluster_assignments == cluster, :]
            self.cluster_centers[cluster] = np.mean(cluster_points, axis=0)

        
    def score(self, X):
        """
        Calculates sum-of-squared-errors (aka inertia aka within-cluster variance)
        for the points in X.  The model has to be fitted before you can call this method.
        This methods calls transform(X) to assign the points.  It should
        then calculate the sum of the squared Euclidean distance between every point and
        its assigned cluster center.
        """
        pass
        
        
    def fit(self, X):
        """
        Fits the model.
        """
        self.initialize_clusters(X)
        for i in range(self.max_iter):
            assignments, distances = self.assign_points(X)
            assignments = self.reinitialize_empty_clusters(X, assignments, distances)
            self.update_centers(X, assignments)
        
    def predict(self, X):
        """
        Assigns the given points to a previously fit model.
        """
        assignments, _ = self.assign_points(X)
        return assignments
        
    def fit_predict(self, X):
        """
        Fits the model and then assigns the points.
        """
        self.fit(X)
        return self.predict(X)
        
        
        
        
        
        