import numpy, pandas

class SLR_GD:
    """
    This SLR class uses Stochastic GD to
    determine the values of B0 and B1.
    
    I know GD should not be used here, but 
    just to learn - how it works, I did.
    
    This is just for single X and Y.
    
    Example
    -------
    >>> model = SLR_GD(x, y)
    >>> model.calculate(epochs=20, learning_rate=0.001)
    >>> model.coef
    >>> model.intecept
    >>> model.predict(x)
    
    """
    def __init__(self, x, y):
        self.x = numpy.array(x)
        self.y = numpy.array(y)
        self.model = lambda B0, B1, xi: B0 + B1 * xi
        
    def calculate(self, epochs: int, learning_rate: float):
        epochs = epochs
        B0 = B1 = 0.0
        
        self._B0s = []; self._B1s = []; self.errors = []
        for i in range(epochs):
            for xi, yi in zip(self.x, self.y):
                pred = self.model(B0, B1, xi)
                error = pred - yi
                B0 = B0 - learning_rate * error
                B1 = B1 - learning_rate * error * xi
                
                self._B0s.append(B0)
                self._B1s.append(B1)
                self.errors.append(error)
        
        self.results = pandas.DataFrame({"B0": self._B0s,
                                    "B1": self._B1s,
                                    "errors": self.errors})
        
        min_id = self.results.errors.apply(abs).idxmin()
        self.intercept = self.results.iloc[min_id].B0
        self.coef = self.results.iloc[min_id].B1
                
    def predict(self, x):
        try:
            y_hat = []
            for xi in x:
                y_hat.append(self.model(self.intercept, self.coef, xi))
            self._y_hat = numpy.array(y_hat)
            return self._y_hat
        except ValueError:
            raise Exception("Please run `calculate` method first - then predict")
            
    def RMSE(self):
        try: return numpy.sqrt(((self._y_hat - self.y) ** 2).mean())
        except:
            raise Exception("Please run `predict` method first - then try this")
        