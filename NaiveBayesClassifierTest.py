import unittest
import numpy as np

from Models.NaiveBayesClassifier import NaiveBayesClassifier


class NaiveBayesTests(unittest.TestCase):

    def setUp(self):
        # Подготовим данные для тестов
        self.X = np.array([
            [1, 'S'], [1, 'M'], [1, 'M'], [1, 'S'], [1, 'S'],
            [2, 'S'], [2, 'M'], [2, 'M'], [2, 'L'], [2, 'L'],
            [3, 'L'], [3, 'M'], [3, 'M'], [3, 'L'], [3, 'L']
        ])

        self.y = np.array([1, 1, 1, -1, -1, -1, -1, 1, -1, 1, 1, 1, 1, 1, -1])

        # Создаем два классификатора: один без сглаживания, другой со сглаживанием
        self.classifier_no_smoothing = NaiveBayesClassifier(laplace_smoothing=False)
        self.classifier_with_smoothing = NaiveBayesClassifier(laplace_smoothing=True)
        
        # Обучаем модели
        self.classifier_no_smoothing.fit(self.X, self.y)
        self.classifier_with_smoothing.fit(self.X, self.y)

    def test_prediction_without_smoothing(self):
        # Тестируем предсказание без Лапласовского сглаживания
        test_point = [2, 'S']
        prediction = self.classifier_no_smoothing.predict(test_point)
        self.assertNotEqual(prediction, -1, f"Prediction without Laplace smoothing failed. {test_point}")

    def test_prediction_with_smoothing(self):
        # Тестируем предсказание с Лапласовским сглаживанием
        test_point = [2, 'S']
        prediction = self.classifier_with_smoothing.predict(test_point)
        self.assertEqual(prediction, -1, f"Prediction with Laplace smoothing failed. {test_point}")


if __name__ == '__main__':
    unittest.main()
