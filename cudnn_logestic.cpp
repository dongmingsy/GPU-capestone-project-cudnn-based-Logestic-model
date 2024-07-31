LogisticModel::LogisticModel() {
	// Initialize cuDNN
	cudnnCreate(&cudnn);

	// Initialize tensor descriptors
	cudnnCreateTensorDescriptor(&input_desc);
	cudnnCreateTensorDescriptor(&output_desc);

	// Initialize convolution descriptor
	cudnnCreateConvolutionDescriptor(&conv_desc);

	// Initialize activation descriptor
	cudnnCreateActivationDescriptor(&activation_desc);
	cudnnSetActivationDescriptor(activation_desc, CUDNN_ACTIVATION_SIGMOID, CUDNN_PROPAGATE_NAN, 0.0);

	// Allocate device memory for input, output, and filter tensors
	cudaMalloc(&d_input, input_size * sizeof(float));
	cudaMalloc(&d_output, num_classes * sizeof(float));
	cudaMalloc(&d_filter, filter_size * sizeof(float));
}

LogisticModel::~LogisticModel() {
	// Destroy descriptors
	cudnnDestroyTensorDescriptor(input_desc);
	cudnnDestroyTensorDescriptor(output_desc);
	cudnnDestroyConvolutionDescriptor(conv_desc);
	cudnnDestroyActivationDescriptor(activation_desc);

	// Free device memory
	cudaFree(d_input);
	cudaFree(d_output);
	cudaFree(d_filter);

	// Destroy cuDNN handle
	cudnnDestroy(cudnn);
}

void LogisticModel::train(const std::vector<std::vector<float>>& data, const std::vector<float>& labels, int epochs, float learning_rate) {
    int batch_size = 1; // For simplicity, we use a batch size of 1
    float alpha = 1.0f, beta = 0.0f;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        for (size_t i = 0; i < data.size(); ++i) {
            // Copy input data to device
            cudaMemcpy(d_input, data[i].data(), input_size * sizeof(float), cudaMemcpyHostToDevice);

            // Forward pass
            cudnnConvolutionForward(cudnn, &alpha, input_desc, d_input, filter_desc, d_filter, conv_desc, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, nullptr, 0, &beta, output_desc, d_output);
            cudnnActivationForward(cudnn, activation_desc, &alpha, output_desc, d_output, &beta, output_desc, d_output);

            // Compute loss and gradients (simplified for logistic regression)
            float h_output[num_classes];
            cudaMemcpy(h_output, d_output, num_classes * sizeof(float), cudaMemcpyDeviceToHost);
            float loss = 0.0f;
            for (int j = 0; j < num_classes; ++j) {
                float label = (labels[i] == j) ? 1.0f : 0.0f;
                loss += (h_output[j] - label) * (h_output[j] - label);
            }
            loss /= 2.0f;

            // Backward pass (simplified for logistic regression)
            float h_grad_output[num_classes];
            for (int j = 0; j < num_classes; ++j) {
                float label = (labels[i] == j) ? 1.0f : 0.0f;
                h_grad_output[j] = h_output[j] - label;
            }
            cudaMemcpy(d_output, h_grad_output, num_classes * sizeof(float), cudaMemcpyHostToDevice);

            cudnnActivationBackward(cudnn, activation_desc, &alpha, output_desc, d_output, output_desc, d_output, output_desc, d_output, &beta, output_desc, d_output);
            cudnnConvolutionBackwardFilter(cudnn, &alpha, input_desc, d_input, output_desc, d_output, conv_desc, CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0, nullptr, 0, &beta, filter_desc, d_filter);

            // Update weights
            float h_filter[input_size * num_classes];
            cudaMemcpy(h_filter, d_filter, input_size * num_classes * sizeof(float), cudaMemcpyDeviceToHost);
            for (int j = 0; j < input_size * num_classes; ++j) {
                h_filter[j] -= learning_rate * h_filter[j];
            }
            cudaMemcpy(d_filter, h_filter, input_size * num_classes * sizeof(float), cudaMemcpyHostToDevice);
        }
    }
}

int LogisticModel::predict(const std::vector<float>& input_data) {
    float alpha = 1.0f, beta = 0.0f;

    // Copy input data to device
    cudaMemcpy(d_input, input_data.data(), input_size * sizeof(float), cudaMemcpyHostToDevice);

    // Forward pass
    cudnnConvolutionForward(cudnn, &alpha, input_desc, d_input, filter_desc, d_filter, conv_desc, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, nullptr, 0, &beta, output_desc, d_output);
    cudnnActivationForward(cudnn, activation_desc, &alpha, output_desc, d_output, &beta, output_desc, d_output);

    // Retrieve the output and determine the predicted class
    float h_output[num_classes];
    cudaMemcpy(h_output, d_output, num_classes * sizeof(float), cudaMemcpyDeviceToHost);

    int predicted_class = std::distance(h_output, std::max_element(h_output, h_output + num_classes));
    return predicted_class;
}
