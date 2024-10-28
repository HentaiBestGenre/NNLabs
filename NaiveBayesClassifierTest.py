import unittest
import numpy as np
from Models.NaiveBayesClassifier import NaiveBayesClassifier


class NaiveBayesTests(unittest.TestCase):

    def setUp(self):
        self.X = np.array([
            [1, 'S'], [1, 'M'], [1, 'M'], [1, 'S'], [1, 'S'],
            [2, 'S'], [2, 'M'], [2, 'M'], [2, 'L'], [2, 'L'],
            [3, 'L'], [3, 'M'], [3, 'M'], [3, 'L'], [3, 'L']
        ])
        self.y = np.array([1, 1, 1, -1, -1, -1, -1, 1, -1, 1, 1, 1, 1, 1, -1])

        self.classifier_no_smoothing = NaiveBayesClassifier(laplace_smoothing=False)
        self.classifier_with_smoothing = NaiveBayesClassifier(laplace_smoothing=True)
        
        self.classifier_no_smoothing.fit(self.X, self.y)
        self.classifier_with_smoothing.fit(self.X, self.y)

    def test_prediction_without_smoothing(self):
        test_points = [
            ([2, 'S'], "Test prediction without smoothing for point [2, 'S']"),
            ([3, 'L'], "Test prediction without smoothing for point [3, 'L']"),
            ([1, 'L'], "Test prediction without smoothing for point [1, 'L']")
        ]
        
        for test_point, message in test_points:
            with self.subTest(msg=message):
                prediction = self.classifier_no_smoothing.predict(test_point)
                predicted_class = max(prediction, key=prediction.get)
                print(f"{test_point} - Prediction without Laplace smoothing: {predicted_class}. Rates: {prediction}")

    def test_prediction_with_smoothing(self):
        test_points = [
            ([2, 'S'], "Test prediction with smoothing for point [2, 'S']"),
            ([3, 'L'], "Test prediction with smoothing for point [3, 'L']"),
            ([1, 'L'], "Test prediction with smoothing for point [1, 'L']")
        ]
        
        for test_point, message in test_points:
            with self.subTest(msg=message):
                prediction = self.classifier_with_smoothing.predict(test_point)
                predicted_class = max(prediction, key=prediction.get)
                print(f"{test_point} - Prediction with Laplace smoothing: {predicted_class}. Rates: {prediction}")


if __name__ == '__main__':
    unittest.main()
