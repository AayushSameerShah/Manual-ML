
class LDA:
    """
    This class is intended to give the hands on LDA.
    
    Note:
    → Only one feature x is supported.
    → Give labels with features as it is supervised.
    
    Good:
    → Available with built in prediction
    → Available with accuracy method.
    
    How:
    >>> model = LDA(x, y)
    >>> model.predict(new_x)
    >>> model.accuracy(real_y)
    
    """
    
    def __init__(self, x, y):
        self._df = pd.DataFrame({"x":x, "y":y})
        self._means = self._get_mean()
        self._variance = self._get_variance()
        self._probs = self._get_probs()
    
    def _get_mean(self):
        return self._df.groupby('y').x.mean()

    def _get_variance(self):
        squared_difference = self._df.groupby("y").apply(lambda x: ((x.x - x.x.mean()) ** 2).sum())
        E_squared_difference = squared_difference.sum()
        n = len(self._df.x)
        k = self._df.y.nunique()
        return E_squared_difference / (n - k)

    def _get_probs(self):
        counts = self._df.y.value_counts()
        return counts / sum(counts)
    
    def predict(self, xi):
        disc_by_class = {}
        for class_ in self._df.y.unique():
            mean = self._means.loc[class_]
            prob = self._probs.loc[class_]
            variance = self._variance
            disc = xi * (mean / variance ** 2) - (mean ** 2 / (2 * variance ** 2)) + np.log(prob)
            disc_by_class[class_] = disc
            
        pred_df = pd.DataFrame(disc_by_class)
        self._new_df = pd.concat([xi, pred_df], axis=1)
        self._new_df["Predicted_class"] = self._new_df.apply(lambda row:row[1:].argmax(), axis=1)
        return self._new_df
    
    def accuracy(self, real_y):
        if hasattr(self, '_new_df'):
            real_y = np.array(real_y)
            if len(real_y) == len(self._new_df.Predicted_class):
                right_preds = (real_y == self._new_df.Predicted_class ).sum()
                acc = (right_preds / self._new_df.shape[0]) * 100
                return str(round(acc, 2)) + " %"
            else: print("Size mismatched. Please provide `true_y` same as the length of provided features' length.")
        else: print("Please predict first. Then run this method")
