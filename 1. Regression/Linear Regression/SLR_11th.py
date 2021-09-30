import numpy

class SLR_11th:
    '''
    This is a class for SLR
    Implementation of the simplest formula learnt in 11th
    
    No gredient descent is used.
    
    Usage
    -----
    >>> model = SLR_11th(x, y)
    >>> model.coef
    >>> model.intercept
    
    Note
    ----
    The x and y must be iterable - list, tuple, array, series. 
    '''
    def __init__(self, x, y):
        self.x = numpy.array(x)
        self.y = numpy.array(y)
        
        self.coef = self._calculate_coef()
        self.intercept = self._calculate_inter()
        
    def _calculate_coef(self):
        self._x_mean = self.x.mean()
        self._y_mean = self.y.mean()

        self._xi_xMean = self.x - self._x_mean
        self._yi_yMean = self.y - self._y_mean
        
        self._xi_xM_x_yi_yM = self._xi_xMean * self._yi_yMean
        self._xi_xM2 = self._xi_xMean ** 2
        return self._xi_xM_x_yi_yM.sum() / self._xi_xM2.sum()
    
    def _calculate_inter(self):
        return self._y_mean - self.coef * self._x_mean
    
    def RMSE(self):
        predict = lambda x: self.intercept + self.coef * x
        self._y_hat = numpy.array(list(map(predict, self.x)))
        return numpy.sqrt(((self._y_hat - self.y) ** 2).mean())
