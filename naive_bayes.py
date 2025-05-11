import pandas as pd
import numpy as np


class NaiveBayes:
    def __init__(self):
        self.priors = {}
        self.likelihoods = {}
        self.classes = None
        self.feature_info = {}

    def fit(self, X, y):
        # Get unique classes from target variable
        self.classes = y.unique()
        n_samples = len(y)

        # Calculate class priors: P(class) = count(class) / total_samples
        for cls in self.classes:
            self.priors[cls] = np.sum(y == cls) / n_samples

        # Calculate likelihoods for each feature
        for feature in X.columns:
            # Store feature metadata
            self.feature_info[feature] = {
                'n_categories': X[feature].nunique(),  # Number of unique values
                'unique_vals': X[feature].unique()     # Actual unique values
            }
            self.likelihoods[feature] = {}

            # Calculate P(feature=value|class) for each class
            for cls in self.classes:
                # Filter data for current class
                cls_mask = (y == cls)
                feature_counts = X[feature][cls_mask].value_counts()
                total_cls_samples = np.sum(cls_mask)

                # Apply Laplace smoothing: (count + 1) / (class_samples + num_categories)
                self.likelihoods[feature][cls] = {
                    val: (feature_counts.get(val, 0) + 1) /
                         (total_cls_samples + self.feature_info[feature]['n_categories'])
                    for val in self.feature_info[feature]['unique_vals']
                }

    def predict(self, X):
        predictions = []
        # Iterate through each test instance
        for _, row in X.iterrows():
            max_prob = -np.inf  # Initialize with negative infinity
            best_class = None

            # Calculate probability for each class
            for cls in self.classes:
                # Start with class prior probability (in log space)
                class_prob = np.log(self.priors[cls])

                # Accumulate feature likelihoods
                for feature in X.columns:
                    val = row[feature]
                    # Handle unseen feature values with uniform probability
                    if val not in self.feature_info[feature]['unique_vals']:
                        prob = 1 / self.feature_info[feature]['n_categories']
                    else:
                        prob = self.likelihoods[feature][cls][val]

                    # Add log-likelihood to avoid multiplying small numbers
                    class_prob += np.log(prob)

                # Update best class if current probability is higher
                if class_prob > max_prob:
                    max_prob = class_prob
                    best_class = cls

            predictions.append(best_class)
        return np.array(predictions)

if __name__ == '__main__':
    # Load raw data
    train_df = pd.read_csv('mushroom_train.data', header=None)
    test_df = pd.read_csv('mushroom_test.data', header=None)

    def encode_data(df):
        encoded = df.copy()
        encoders = {}
        for col in df.columns:
            encoded[col], encoders[col] = df[col].factorize()
        return encoded, encoders

    # Encode training and test data
    train_encoded, _ = encode_data(train_df)
    test_encoded, _ = encode_data(test_df)

    # Split into features (X) and target (y)
    # Assumes first column is target, rest are features
    X_train = train_encoded.iloc[:, 1:]  # Features (columns 1-end)
    y_train = train_encoded.iloc[:, 0]   # Target (column 0)
    X_test = test_encoded.iloc[:, 1:]
    y_test = test_encoded.iloc[:, 0]

    # Initialize and train model
    nb = NaiveBayes()
    nb.fit(X_train, y_train)

    # Make predictions and calculate accuracy
    y_pred = nb.predict(X_test)
    accuracy = np.mean(y_pred == y_test.values)
    print(f"Custom Naive Bayes Accuracy: {accuracy:.4f}")