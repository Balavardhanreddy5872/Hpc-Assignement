import csv
import math
import cProfile
import pstats
import numpy as np
import time
from memory_profiler import profile
import psutil
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

def load_data(file_path):
    data = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file, delimiter=';')
        header_skipped = False
        for row in reader:
            if not header_skipped:
                header_skipped = True
                continue
            data.append(list(map(float, row)))
    return data

def euclidean_distance(point1, point2):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(point1, point2)))

def get_neighbors(training_data, test_instance, k):
    distances = [(i, euclidean_distance(test_instance, row[:-1])) for i, row in enumerate(training_data)]
    distances.sort(key=lambda x: x[1])
    return [i for i, _ in distances[:k]]

def classify(neighbors, labels):
    votes = [labels[i] for i in neighbors]
    return max(set(votes), key=votes.count)

def k_nearest_neighbor_classifier(training_data, test_data, train_labels, k):
    predictions = []
    for test_instance in test_data:
        neighbors = get_neighbors(training_data, test_instance, k)
        predictions.append(classify(neighbors, train_labels))
    return predictions

def calculate_accuracy(predictions, actual_labels):
    correct_predictions = sum(1 for p, a in zip(predictions, actual_labels) if p == a)
    total_predictions = len(predictions)
    accuracy = correct_predictions / total_predictions
    return accuracy

def svm_classifier(X_train, y_train, X_test, y_test, C_values):
    cross_val_scores = []
    test_accuracies = []

    for C in C_values:
        svm = SVC(kernel='rbf', C=C)
        
        # Cross-validation
        scores = cross_val_score(svm, X_train, y_train, cv=3)
        mean_score = np.mean(scores)
        cross_val_scores.append(mean_score)

        # Train the SVM model with the current C value
        svm.fit(X_train, y_train)
        test_predictions = svm.predict(X_test)
        test_accuracy = calculate_accuracy(test_predictions, y_test)
        test_accuracies.append(test_accuracy)

    return cross_val_scores, test_accuracies

def random_forest_classifier(X_train_rf, y_train_rf, X_test_rf, y_test, n_estimators_rf):
    cross_val_scores_rf = []
    test_accuracies_rf = []

    for n_estimators in n_estimators_rf:
        rf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        
        # Cross-validation
        scores = cross_val_score(rf, X_train_rf, y_train_rf, cv=3)
        mean_score = np.mean(scores)
        cross_val_scores_rf.append(mean_score)

        # Train the Random Forest model with the current number of estimators
        rf.fit(X_train_rf, y_train_rf)
        test_predictions_rf = rf.predict(X_test_rf)
        test_accuracy_rf = accuracy_score(test_predictions_rf, y_test)
        test_accuracies_rf.append(test_accuracy_rf)

    return cross_val_scores_rf, test_accuracies_rf

@profile
def profile_knn_svm_rf():
    # Load data
    data = load_data('winequality-white-Train.csv')
    test_data = load_data('winequality-white-Test.csv')

    # Extract features and labels
    features = np.array([row[:-1] for row in data])
    labels = np.array([int(row[-1]) for row in data])
    test_features = np.array([row[:-1] for row in test_data])
    test_labels = np.array([int(row[-1]) for row in test_data])

    # Profiling setup
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Measure execution time
    start_time = time.time()

    # Implement 1-NNC and report accuracy
    k_1_predictions = k_nearest_neighbor_classifier(features, test_features, labels, k=1)
    accuracy_1NNC = calculate_accuracy(k_1_predictions, test_labels)
    print("Accuracy of 1-NNC:", accuracy_1NNC)

    # Implement 3-NNC and report accuracy
    k_3_predictions = k_nearest_neighbor_classifier(features, test_features, labels, k=3)
    accuracy_3NNC = calculate_accuracy(k_3_predictions, test_labels)
    print("Accuracy of 3-NNC:", accuracy_3NNC)

    # SVM
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(test_features)  # Use the same scaler for test data

    # Define a range of C values to be tested
    C_values = [0.01, 0.1, 1, 10, 100]

    # SVM Classifier
    cross_val_scores, test_accuracies = svm_classifier(X_train, y_train, X_test, test_labels, C_values)

    # Print SVM results
    for i in range(len(C_values)):
        print(f"{i}) C:{C_values[i]} {'':<5} Cross-Validation Score:{cross_val_scores[i]:.4f} Test Accuracy:{test_accuracies[i]:.4f}")

    # Random Forest Classifier
    n_estimators_rf = [50, 100, 150, 200, 250]
    cross_val_scores_rf, test_accuracies_rf = random_forest_classifier(features, labels, test_features, test_labels, n_estimators_rf)

    # Print Random Forest results
    for i in range(len(n_estimators_rf)):
        print(f"{i}) Estimators:{n_estimators_rf[i]} {'':<5} Cross-Validation Score:{cross_val_scores_rf[i]:.4f} Test Accuracy:{test_accuracies_rf[i]:.4f}")

    # Measure execution time
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Total Execution Time: {execution_time:.4f} seconds")

    # Measure memory usage
    process = psutil.Process()
    memory_usage = process.memory_info().rss / 1024 / 1024  # in megabytes
    print(f"Memory Usage: {memory_usage:.4f} MB")

    profiler.disable()

    # Save profiling results to a file
    with open('knn_svm_rf_profiling_results.txt', 'w') as result_file:
        stats = pstats.Stats(profiler, stream=result_file)
        stats.sort_stats('cumulative')
        stats.print_stats()

# Run the profiling
profile_knn_svm_rf()
