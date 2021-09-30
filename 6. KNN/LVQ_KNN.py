
import math
import numpy as np
from KNN import KNN


class LVQ_KNN:
    """
    This class is implemented to build LVQ model.
    It gives a nice range of hyperparameters to tune
    and works much faster than the traditional KNN.
    
    For the prediction part, it uses KNN though, but
    with the less number of layers, it works amazingly fast.
    
    How To
    ------
        # Prepare X and y
    >>> X = iris.drop("species", axis=1)
    ... y = iris["species"]
    
        # Train
    >>> model = LVQ_KNN(X, y, n_layers=10, 
                        learning_rate=0.3, epochs=200,
                        learning_mode="static / dynamic")
    >>> model.predict(X)                              
    """
    def __init__(self, X, y, n_layers, learning_rate=0.3, epochs=2,
                 learning_mode="dynamic"):
        self.X = np.array(X); self.y = np.array(y)
        self.n_layers = n_layers
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.learning_mode = learning_mode
        
        if X.ndim != 2:
            raise NotImplementedError(\
            """
            The dimention of the features 
            must be 2D.
            """)
        if (len(X) != len(y)) or (y.ndim != 1):
            raise NotImplementedError(\
            """
            The length of features 
            and target mismatched.
            """)
        if n_layers > len(self.X):
            raise NotImplementedError(\
            """
            `n_layers` must be less than
            total rows in dataset.
            """)
            
        self.NN = self.randomize_init()
        for epoch in range(self.epochs):
            # LearningRate = LearningRate × (1 − 𝑛𝑡ℎEpoch / TotalEpochs)
            if self.learning_mode == "dynamic":
                self.learning_rate = self.learning_rate * (1 - epoch / self.epochs)
            self.compress_data()
        
        
    def randomize_init(self):
        shuffled_X = None
        for column in range(self.X.shape[1]):
            mask = np.random.permutation(self.X.shape[0])[:self.n_layers]
            if shuffled_X is None:
                shuffled_X = self.X[mask, column]
            else:
                shuffled_X = np.c_[shuffled_X, self.X[mask, column]]
        self.stratified_y_column = self.stratified_sampling()[: self.n_layers]
        return shuffled_X
     
        
    def compress_data(self):
        for row in self.X:
            distances = self.euclidean_distance(row, self.NN)
            bmu_at = distances.argmin()
            bmu = self.NN[bmu_at]
            if self.y[bmu_at] == self.stratified_y_column[bmu_at]:
                new_bmu = bmu + self.learning_rate * (row - bmu)
            else:
                new_bmu = bmu - self.learning_rate * (row - bmu)
            self.NN[bmu_at] = new_bmu
    
    
    def predict(self, X, k):
        model = KNN(self.NN, self.stratified_y_column)
        return model.predict(X, k=k)
    
    
    @staticmethod
    def euclidean_distance(A, B):
        return ((A - B) ** 2).sum(1) ** 0.5
    
    
    def stratified_sampling(self):
        total = len(self.y)
        new_y = []
        for class_, count in zip(*np.unique(self.y, return_counts=True)):
            new_count = math.ceil((count * self.n_layers) / total)
            new_y.extend(np.full(new_count, class_))
        return new_y
