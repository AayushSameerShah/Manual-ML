import numpy as np
import pandas as pd

class NaiveBayes():
    """
    This class implements the categorical naive bayes algorithm
    which requires all features and the targets to be discrete.
    
    The performance and predictions are too similar to sklearn's
    CategoricalNB model. Accuracy is so similar in comparision.
    
    How To
    ------
    >>> model = NaiveBayes(wholeDF, "target_y")
    >>> model.predict(wholeDF)
        
        # Concatinating to see the predicted probabilities
    >>> pred = pd.concat([wholeDF, pd.DataFrame(model.predict(wholeDF))], axis=1)
    
        # Checking for accuracy
    >>> (pred.y == pred.pred).sum() / pred.shape[0] * 100

    
    """
    def __init__(self, data: pd.DataFrame, target: str):
        
        self.data = data
        self.target = target
        
        if not self.target in self.data:
            raise NotImplementedError("Target provided is not found in the data.")
        
        self.class_counts = self.data[self.target].value_counts()
        self.class_probs = self.class_counts / self.data.shape[0]

        for_each_feature = []
        for feature in self.data.drop(self.target, axis=1):
            for_each_feature.append(self.calculate_conditional_prob(feature))

        self.lookup = pd.concat(for_each_feature).unstack()
    
    
    def calculate_conditional_prob(self, feature):
        probs = {}
        for category in self.data[feature].unique():
            for class_ in self.data[self.target].unique():
                filter_ = (self.data[feature] == category) & (self.data[self.target] == class_)
                A_n_B = self.data[filter_].shape[0]
                B = self.class_counts[class_]
                probs[(category, class_)] = A_n_B / B
        return pd.Series(probs)
    
    
    def predict(self, data):
        each = []
        for features in data.drop(self.target, axis=1).values:
            probs = {}
            for class_ in self.lookup.columns:
                multi = self.lookup.loc[features, class_].prod()
                multi *= self.class_probs[class_]
                probs[class_] = round(multi, 5)
            winner = sorted(probs.items(), key=lambda i:i[1], reverse=True)[0][0]
            probs["pred"] = winner
            each.append(probs)
        return each
