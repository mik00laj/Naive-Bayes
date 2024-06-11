import numpy as np
class NaiveBayes:
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)

       # Obliczanie parametrów
        self.mean = np.zeros((n_classes, n_features), dtype=np.float64)  # Średnia
        self.var = np.zeros((n_classes, n_features), dtype=np.float64)   # Wariancja
        self.priors = np.zeros(n_classes, dtype=np.float64)                     # Apprioryczne prawdopodobieństwo dla każdej klasy


        for idx, c in enumerate(self.classes):
            X_c = X[y == c]
            self.mean[idx, :] = X_c.mean(axis=0)
            self.var[idx, :] = X_c.var(axis=0) + 1e-6  # Dodano małą wartość, aby uniknąć dzielenia przez zero
            self.priors[idx] = X_c.shape[0] / float(n_samples)

    # Wariancja cechy może być zerowa, gdy wszystkie próbki w danej klasie mają tę samą wartość dla tej cechy,
    # co oznacza brak różnorodności (wariancji) w tych danych. Dodanie małej wartości zapewnia,
    # że wszystkie wartości wariancji są dodatnie i większe od zera,
    # co umożliwia bezpieczne wykonywanie operacji matematycznych wymaganych w algorytmie Naive Bayes.

    # Predykcja - dla każdej próbki x oblicza się posteriori dla każdej klasy
    # na podstawie uprzednio wyliczonych parametrów (średnia, wariancja, prior).
    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    # Obliczanie prawdopodobieństwa
    # zwraca rozkład prawdopodobieństw dla wszystkich klas
    def predict_proba(self, X):
        y_proba = [self._predict_proba(x) for x in X]
        return np.array(y_proba)

    # Przewidywanie klasy dla danej próbki
    def _predict(self, x):
        posteriors = []

        for idx, c in enumerate(self.classes):
            prior = np.log(self.priors[idx])
            posterior = np.sum(np.log(self._pdf(idx, x)))
            posterior = prior + posterior
            posteriors.append(posterior)

        return self.classes[np.argmax(posteriors)]

    #  przewidywanie prawdopodobieństw posteriori dla różnych klas,
    #  co pozwala modelowi Naive Bayes na przewidywanie najbardziej prawdopodobnej klasy dla nowego przykładu danych.
    def _predict_proba(self, x):
        posteriors = []

        for idx, c in enumerate(self.classes):
            prior = np.log(self.priors[idx])
            posterior = np.sum(np.log(self._pdf(idx, x)))
            posterior = prior + posterior
            posteriors.append(posterior)

        exp_posteriors = np.exp(posteriors)
        return exp_posteriors / np.sum(exp_posteriors)

    # Obliczanie gęstości prawdopodobieństwa
    # Dla każdej cechy w x, obliczana jest gęstość prawdopodobieństwa
    # na podstawie rozkładu normalnego z określoną średnią i wariancją.
    def _pdf(self, class_idx, x):
        mean = self.mean[class_idx]
        var = self.var[class_idx]
        if not np.issubdtype(x.dtype, np.floating):
            x = x.astype(np.float64)
        numerator = np.exp(- (x - mean) ** 2 / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator

