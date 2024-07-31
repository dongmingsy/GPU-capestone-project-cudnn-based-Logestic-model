#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <cudnn.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

std::vector<std::vector<float>> loadCSV(const std::string& filename) {
	std::vector<std::vector<float>> data;
	std::ifstream file(filename);
	std::string line, value;

	while (std::getline(file, line)) {
		std::vector<float> row;
		std::stringstream ss(line);
		while (std::getline(ss, value, ',')) {
			row.push_back(std::stof(value));
		}
		data.push_back(row);
	}
	return data;
}

// Function to calculate confusion matrix
cv::Mat calculateConfusionMatrix(const std::vector<int>& true_labels, const std::vector<int>& predicted_labels, int num_classes) {
    cv::Mat confusion_matrix = cv::Mat::zeros(num_classes, num_classes, CV_32S);
    for (size_t i = 0; i < true_labels.size(); ++i) {
        confusion_matrix.at<int>(true_labels[i], predicted_labels[i])++;
    }
    return confusion_matrix;
}

// Function to calculate ROC curve data
void calculateROC(const std::vector<int>& true_labels, const std::vector<float>& predicted_probs, int num_classes) {
    for (int c = 0; c < num_classes; ++c) {
        std::vector<float> tpr, fpr;
        for (float threshold = 0.0; threshold <= 1.0; threshold += 0.01) {
            int tp = 0, fp = 0, fn = 0, tn = 0;
            for (size_t i = 0; i < true_labels.size(); ++i) {
                bool positive = predicted_probs[i * num_classes + c] >= threshold;
                if (true_labels[i] == c) {
                    if (positive) tp++;
                    else fn++;
                } else {
                    if (positive) fp++;
                    else tn++;
                }
            }
            tpr.push_back(static_cast<float>(tp) / (tp + fn));
            fpr.push_back(static_cast<float>(fp) / (fp + tn));
        }
        std::cout << "Class " << c << " ROC Curve Data:" << std::endl;
        for (size_t i = 0; i < tpr.size(); ++i) {
            std::cout << "Threshold: " << i * 0.01 << " TPR: " << tpr[i] << " FPR: " << fpr[i] << std::endl;
        }
    }
}

int main() {
    std::string filename = "iris_preprocessed.csv";
    auto data = loadCSV(filename);
    std::vector<float> labels;
    preprocessData(data, labels);

    int input_size = data[0].size();
    int num_classes = 3;

    // Split data into training and testing sets
    std::vector<std::vector<float>> train_data, test_data;
    std::vector<float> train_labels, test_labels;

    // Shuffle data
    std::vector<int> indices(data.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);

    // Split 80% for training and 20% for testing
    size_t train_size = static_cast<size_t>(0.8 * data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        if (i < train_size) {
            train_data.push_back(data[indices[i]]);
            train_labels.push_back(labels[indices[i]]);
        } else {
            test_data.push_back(data[indices[i]]);
            test_labels.push_back(labels[indices[i]]);
        }
    }

    // Train the model
    LogisticModel model(input_size, num_classes);
    model.train(train_data, train_labels, 1000, 0.01f);

    // Evaluate the model
    std::vector<int> true_labels;
    std::vector<int> predicted_labels;
    std::vector<float> predicted_probs;
    for (size_t i = 0; i < test_data.size(); ++i) {
        float prediction = model.predict(test_data[i]);
        true_labels.push_back(static_cast<int>(test_labels[i]));
        predicted_labels.push_back(static_cast<int>(prediction));
        // Assuming model.predict returns probabilities for each class
        std::vector<float> probs = model.predict_proba(test_data[i]);
        predicted_probs.insert(predicted_probs.end(), probs.begin(), probs.end());
    }

    // Calculate and print confusion matrix
    cv::Mat confusion_matrix = calculateConfusionMatrix(true_labels, predicted_labels, num_classes);
    std::cout << "Confusion Matrix:" << std::endl;
    std::cout << confusion_matrix << std::endl;

    // Calculate and print ROC curve data
    calculateROC(true_labels, predicted_probs, num_classes);

    return 0;
}