class LogisticModel {
public:
    LogisticModel(int input_size, int num_classes);
    ~LogisticModel();
    void train(const std::vector<std::vector<float>>& data, const std::vector<float>& labels, int epochs, float learning_rate);
    float predict(const std::vector<float>& input);

private:
    cudnnHandle_t cudnn;
    cudnnTensorDescriptor_t input_desc, output_desc;
    cudnnFilterDescriptor_t filter_desc;
    cudnnConvolutionDescriptor_t conv_desc;
    cudnnActivationDescriptor_t activation_desc;
    float* d_input;
    float* d_output;
    float* d_filter;
    int input_size;
    int num_classes;
};