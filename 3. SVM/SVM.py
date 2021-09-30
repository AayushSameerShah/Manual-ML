import numpy as np

class SVM:
    """
    This class implement WITH B0
    
    How To
    ------
        # Prepare X and y
    >>> X = df[["x1", "x2"]]
    ... y = df["y"]
    
        # Train
    >>> model = SVM(X, y, learning_rate=0.5, acc_per_epoch=True)
    >>> model.accuracies

    """
    
    def __init__(self, X, y, learning_rate=0.001, epochs=10, acc_per_epoch=False):
        # Normal conversion
        self.X = np.array(X)
        self.y = np.array(y)
        
        # Usual Checking
        if self.X.ndim != 2:
            raise NotImplementedError("The features must be 2D")
        if len(self.X) != len(self.y):
            raise NotImplementedError("Please! Come on!")
            
        # Initialization of Betas
        self.Bi = np.array([0.0] * self.X.shape[1])
        self.B0 = 0.0
        
        # The learning rate and epoch setting
        self.learning_rate = learning_rate
        self.epochs = epochs
        
        # This ↓ will contain the accuracies if it has to be
        # found with `acc_per_epoch=True`.
        self.accuracies = []
        
        # | keeps track of th iteration 
        # ↓ which will be passed in the formulae
        th_iteration = 1
        
        # For each th epoch
        for th_epoch in range(self.epochs):
            # for each individual row
            for th_row in range(len(self.X)):
                # first we will calculate the initial output
                otpt = self.output(th_row)
                # 2 different functions to update weights
                if otpt > 1:
                    self.Bi = self.output_greater_than_1_Bi(th_iteration) # for Bi
                    self.B0 = self.output_greater_than_1_B0(th_iteration) # for B0

                else:
                    self.Bi = self.output_less_than_1_Bi(th_iteration, th_row) # for Bi
                    self.B0 = self.output_less_than_1_B0(th_iteration, th_row) # for B0
                th_iteration += 1
            
            # If we are told to find the accuracies
            if acc_per_epoch:
                outputs = self.predict(self.X)
                acc = (self.y == self.crisp(outputs)).sum() / len(self.y)
                self.accuracies.append(acc)
    
    # Will return the value which then will be checked 
    def output(self, th_row):
        return self.y[th_row] * (self.Bi * self.X[th_row]).sum() + self.B0
    
    
    def output_less_than_1_Bi(self, th_iteration, th_row):
        return ((1 - (1 / th_iteration)) * self.Bi + (1 / (self.learning_rate * th_iteration)) * (self.y[th_row] * self.X[th_row]))
    
    
    def output_greater_than_1_Bi(self, th_iteration):
        return (1 - (1 / th_iteration)) * self.Bi
    
    
    def output_less_than_1_B0(self, th_iteration, th_row):
        return ((1 - (1 / th_iteration)) * self.B0 + (1 / (self.learning_rate * th_iteration)) * (self.y[th_row]))
    
    
    def output_greater_than_1_B0(self, th_iteration):
        return (1 - (1 / th_iteration)) * self.B0
    
    
    def predict(self, X):
        return self.B0 + (self.Bi * X).sum(1)
    
    def crisp(self, outputs):
        pred = []
        for output in outputs:
            if output < 0:
                pred.append(-1)
            else: pred.append(1)
        return pred
