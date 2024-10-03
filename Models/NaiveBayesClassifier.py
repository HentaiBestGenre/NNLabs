from collections import Counter


class NaiveBayesClassifier:
    def __init__(self, laplace_smoothing=False, laplace_lambda=1):
        self.laplace_smoothing = laplace_smoothing
        self.laplace_lambda = laplace_lambda
        self.class_probs = {}
        self.cond_probs = {}

    def fit(self, X, y):
        class_counts = Counter(y)
        total_count = len(y)
        num_classes = len(class_counts)

        if self.laplace_smoothing:
            for cls in class_counts:
                self.class_probs[cls] = (class_counts[cls] + self.laplace_lambda) / (total_count + self.laplace_lambda * num_classes)
        else:
            for cls in class_counts:
                self.class_probs[cls] = class_counts[cls] / total_count
        
        feature_count = len(X[0])
        for cls in class_counts:
            class_indices = [i for i in range(total_count) if y[i] == cls]
            for feature in range(feature_count):
                feature_values = [X[i][feature] for i in class_indices]
                value_counts = Counter(feature_values)
                num_unique_values = len(set(X[:, feature]))
                
                if self.laplace_smoothing:
                    self.cond_probs[(feature, cls)] = {
                        value: (value_counts[value] + self.laplace_lambda) / (len(feature_values) + self.laplace_lambda * num_unique_values)
                        for value in value_counts
                    }
                else:
                    self.cond_probs[(feature, cls)] = {
                        value: value_counts[value] / len(feature_values)
                        for value in value_counts
                    }
    
    def predict(self, x):
        class_scores = {}
        for cls in self.class_probs:
            class_scores[cls] = self.class_probs[cls]
            for feature in range(len(x)):
                if x[feature] in self.cond_probs[(feature, cls)]:
                    class_scores[cls] *= self.cond_probs[(feature, cls)][x[feature]]
                else:
                    if self.laplace_smoothing:
                        class_scores[cls] *= self.laplace_lambda / (len(self.cond_probs[(feature, cls)]) + self.laplace_lambda * len(set(x)))
                    else:
                        class_scores[cls] = 0
        
        return max(class_scores, key=class_scores.get)
