import numpy as np

class LogisticRegression():
    '''
    This is a Logistic Regression from scratch!
    Very easy to use and implement.
    
    Features
    --------
    → It has ability to expand to many features at once. 
      That means, is you can pass n number of features 
      as 2D array and it will work.
      
    → Built in `predict` method that gives predicted labels
    → Built in `accuracies` that will keep track of accuracies
      on each epoch
    → Easy syntax with documentation
    
    How to
    ------
    >>> model = LogisticRegression()
    >>> model.train(epochs=10, learning_rate=0.3)
    >>> model.intercept
    >>> model.coefs
    >>> model.predict(new_x)
    >>> plt.plot(model.accuracies)
    
    Note
    ----
    The model requires data X data in 2D array. Otherwise 
    you will face an error!
    
    '''
    def __init__(self, x, y):
        self.x = np.array(x)
        self.y = np.array(y)
        
        if self.x.ndim  == 1:
            raise ValueError('''Hey! Please insert 2D list / array.
            It's really awkward to work with 1D.
            
            Hint: Use `arr[:, np.newaxis]` to make it 2D if its 1D.''')
        
        if x.shape[0] != len(y):
            raise NotImplementedError(''' Hey! The data length is not matching!!! Kindly match it!''')
        self.intercept = 0.0
        self.coefs = np.zeros(self.x.shape[-1], dtype=np.float32)
    
    def train(self, epochs, learning_rate=0.01):
        self.accuracies = []
        for epoch in range(epochs):
            for X, y in zip(self.x, self.y):
                prediction = self._model(X)
                self.intercept = self.intercept + learning_rate * (y - prediction) * prediction * (1 - prediction)
                for i in range(len(self.coefs)):
                    self.coefs[i] = self.coefs[i] + learning_rate * (y - prediction) * prediction * (1 - prediction) * X[i]

            preds = []
            for X, y in zip(self.x, self.y):
                prediction = self._model(X)
                preds.append(y == self._crisp(prediction))
            self.accuracies.append(self._get_accuracy(preds))
    
    def _model(self, X):
        coefs_add = 0
        for coef, xi in zip(self.coefs, X):
            coefs_add += coef * xi
        return 1 / (1 + np.exp(-(self.intercept + coefs_add)))
    
    def predict(self, new_x):
        new_x = np.array(new_x)
        assert new_x.ndim != 1, "Please insert 2D array"
        assert new_x.shape[-1] == len(self.coefs), "The data is mis-matched."
        labels = []
        for xi in new_x: 
            pred = self._model(xi)
            label = self._crisp(pred)
            labels.append(label)
        return labels
        
    @staticmethod
    def _crisp(pred):
        return 1 if pred > 0.5 else 0
    
    @staticmethod
    def _get_accuracy(preds):
        total_right = sum(preds)
        total_values = len(preds)
        return (total_right / total_values) * 100
