import pandas as pd
import numpy as np


class NaiveBayes:
    def __init__(self):
        self.priors = {}
        self.likelihoods = {}
        self.classes = None
        self.feature_info = {}

    def fit(self, X, y):
        self.classes = y.unique()
        n_samples = len(y)

        for cls in self.classes:
            self.priors[cls] = np.sum(y == cls) / n_samples

        for feature in X.columns:
            self.feature_info[feature] = {
                'n_categories': X[feature].nunique(),
                'unique_vals': X[feature].unique()
            }
            self.likelihoods[feature] = {}

            for cls in self.classes:
                cls_mask = (y == cls)
                feature_counts = X[feature][cls_mask].value_counts()
                total_cls_samples = np.sum(cls_mask)

                self.likelihoods[feature][cls] = {
                    val: ((feature_counts.get(val, 0) + 1) /
                         (total_cls_samples + self.feature_info[feature]['n_categories']))
                    for val in self.feature_info[feature]['unique_vals']
                }

    def predict(self, X):
        predictions = []
        for _, row in X.iterrows():
            max_prob = -np.inf
            best_class = None

            for cls in self.classes:
                class_prob = np.log(self.priors[cls])

                for feature in X.columns:
                    val = row[feature]
                    if val not in self.feature_info[feature]['unique_vals']:
                        # Handle unseen values with uniform probability
                        prob = 1 / self.feature_info[feature]['n_categories']
                    else:
                        prob = self.likelihoods[feature][cls][val]

                    class_prob += np.log(prob)

                if class_prob > max_prob:
                    max_prob = class_prob
                    best_class = cls

            predictions.append(best_class)
        return np.array(predictions)


if __name__ == '__main__':
    train_df = pd.read_csv('mushroom_train.data', header=None)
    test_df = pd.read_csv('mushroom_test.data', header=None)


    def encode_data(df):
        encoded = df.copy()
        encoders = {}
        for col in df.columns:
            encoded[col], encoders[col] = df[col].factorize()
        return encoded, encoders


    train_encoded, _ = encode_data(train_df)
    test_encoded, _ = encode_data(test_df)

    X_train = train_encoded.iloc[:, 1:]
    y_train = train_encoded.iloc[:, 0]
    X_test = test_encoded.iloc[:, 1:]
    y_test = test_encoded.iloc[:, 0]

    nb = NaiveBayes()
    nb.fit(X_train, y_train)
    y_pred = nb.predict(X_test)

    accuracy = np.mean(y_pred == y_test.values)
    print(f"Custom Naive Bayes Accuracy: {accuracy:.4f}")