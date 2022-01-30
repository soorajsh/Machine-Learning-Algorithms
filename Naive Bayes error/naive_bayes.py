import numpy as np
from numpy.core.fromnumeric import mean

class NaiveBayes:
    def fit(self, x, y):
        n_samples, n_features = x.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)


        #init mean, varience and priors 
        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self._var = np.zeros((n_classes, n_features), dtype=np.float64)
        self._priors = np.zeros(n_classes, dtype=np.float64)

        for c in self.classes:
            x_c = x[c==y]
            self._mean[c,:] = x_c.mean(axis=0)
            self._var[c, :] = x_c.var(axis=0)
            self._priors[c ] = x_c.shape[0] / float(n_samples)




    def predict(self, X):
        y_pred = [self._predict(x) for x in X]

    def _predict(self, X):
        posteriors = []

        for idx, c in enumerate(self.classes):
            prior = np.log(self._priors[idx])
            class_conditional = np.sum(np.log(self._probability_density_function(idx, X) ))
            posterior = prior + class_conditional
            posteriors.append(posterior)

            return self.classes[np.argmax(posteriors)]

    def _probability_density_function(self, class_idx, x):
        mean = self._mean[class_idx]
        var = self._var[class_idx]
        numerator = np.exp(- (x-mean)**2 / (2*var))
        denominator = np.sqrt(2 * np.pi* var)
        return numerator/ denominator 



