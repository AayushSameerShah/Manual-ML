
import statistics
import numpy as np

class KNN:
    """
    This model is so great. It also can work on 4 different
    types of distance metrics.
    
    First time, I tried to make a model entirely in numpy.
    Haven't used a single bit of pandas. It makes numpy stronger
    and calculation faster.
    
    
    How To
    ------
    
    >>> model = KNN(X, y)
    >>> pred = model.predict(X, k=3, dis_type="euclidean")
    """
    def __init__(self, X: np.ndarray, y: list):
        X = np.array(X)
        y = np.array(y)
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
        self.stored_X = X
        self.stored_y = y
        
    
    def predict(self, X: np.ndarray, k: int, dis_type="euclidean", p=None):
        X = np.array(X)
        self.dis_type = dis_type
        if (self.dis_type == "minkowaski") and (p == None):
            raise NotImplementedError("Please provide `p` value.")
        self.p = p    
        if X.ndim != 2:
            raise NotImplementedError(\
            """
            The dimention of the features 
            must be 2D.
            """)
        pred_classes = []
        for each_row in X:
            distance = self.get_distance(row=each_row)
            sorted_k_indexes = np.argsort(distance)[:k]
            pred_class = statistics.mode(self.stored_y[sorted_k_indexes])
            pred_classes.append(pred_class)
        return pred_classes
    
    def get_distance(self, row):
        if self.dis_type == "euclidean":
            return ((row - self.stored_X) ** 2).sum(1) ** 0.5
        elif self.dis_type == "manhattan":
            return abs((row - self.stored_X)).sum(1)
        elif self.dis_type == "hamming":
            return abs(row - self.stored_X).sum(1) / len(row)
        elif self.dis_type == "minkowaski":
            return (abs(row - self.stored_X) ** self.p).sum(1) ** (1 / self.p)
        else:
            raise NotImplementedError(\
            f"""
            The distance type chosen is `{self.dis_type}`.
            Please choose from: 
            • euclidean
            • manhattan
            • hamming
            • minkowaski
            """)
