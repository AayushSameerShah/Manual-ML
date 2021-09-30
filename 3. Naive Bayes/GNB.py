
import math
import numpy as np
import pandas as pd

class GaussianNaiveBayes:
    """
    This is an amazing class.
    This class also work with encoded categorical values.
    
    How To
    ------
    >>> model = GaussianNaiveBayes(iris, "species")
    >>> pred = model.predict(iris.drop("species", axis=1))
    >>> pred = pd.DataFrame(pred)
    
        # get accuracy
    >>> (pred.species == pred.winner).sum() / pred.shape[0]
    
    Great!
    """
    def __init__(self, data: pd.DataFrame, target: str):
        self.data = data
        self.target = target
        
        if not self.target in self.data:
            raise NotImplementedError("The given target column is not found in the data.")
            
        self.class_counts = self.data[self.target].value_counts()
        self.class_probs = self.class_counts / self.class_counts.sum()
        self.lookup = self.data.groupby(self.target).agg(["mean", "std"])
    
    
    def predict(self, features):
        feature_names = features.columns
        target = self.data[self.target]

        # For all rows - stores pdfs class wise and winner for that row
        # in the form of dict {0: 0.33, 1: 1.23. winner: 1}
        row_wise_pdf_and_winner = []

        # Iterates through all features (excluding target)
        for row in features.values:
            # Stores the dict {0: 0.33, 1: 1.23. winner: 1}
            # It gets appended in the maind row_wise... ↑ list.
            pdfs_target_wise = {}

            # Iterates through all unique classes
            for target_class in target.unique():
                # Stores all pdfs feature wise. Then we will use
                # it to perform product [0.22, 0.32, 0.12] (in 3 features data)
                pdfs_for_that_target_class = []

                # Maps the feature name with its values to make
                # it easier in lookup
                for column, its_value in zip(feature_names, row):
                    # Returns mean and std from lookup
                    # for that class and feature
                    learned_values = self.lookup.loc[target_class, column]

                    # Appends featurewise pdf in list (then we will multiply)
                    pdfs_for_that_target_class.append(
                        self.pdf(its_value, learned_values["mean"], learned_values["std"]))

                # For that class, we will return the pdf. After multiplying all
                pdfs_target_wise[target_class] = math.prod(pdfs_for_that_target_class) * self.class_probs[target_class]

            pdfs_target_wise["winner"] = sorted(pdfs_target_wise.items(), key=lambda i: i[1], reverse=True)[0][0]
            row_wise_pdf_and_winner.append(pdfs_target_wise)
        return row_wise_pdf_and_winner
    
    @staticmethod
    def pdf(x, mean, std):
        return (1 / (np.sqrt(2 * np.pi) * std)) * np.exp(-(x - mean) ** 2 / (2 * std ** 2))
    
    # -- END of class END -- #