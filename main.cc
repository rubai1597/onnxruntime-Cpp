#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>
#include <opencv2/opencv.hpp>

template <typename T>
std::vector<size_t> argsort(const std::vector<T>& array, bool desc = false);

const std::string TENSOR_DTYPES[] = {
    "UNDEFINED",      "FLOAT32",      "UINT8",          "INT8",
    "UINT16",         "INT16",        "INT32",          "INT64",
    "UINT16",         "INT16",        "INT32",          "INT64",
    "STRING",         "BOOL",         "FLOAT16",        "DOUBLE",
    "UINT32",         "UINT64",       "COMPLEX64",      "COMPLEX128",
    "BFLOAT16",       "FLOAT8E4M3FN", "FLOAT8E4M3FNUZ", "FLOAT8E5M2",
    "FLOAT8E5M2FNUZ",
};

struct TensorInfo {
  std::vector<int64_t> shape;
  ONNXTensorElementDataType dtype;
  int num_elements;
};

template <typename T>
void PrintVectorElements(const std::vector<T>& vec,
                         std::string delimiter = ", ", std::string prefix = "(",
                         std::string postfix = ")");

template <typename T>
void PrintVectorElements(const std::vector<T>& vec, std::string delimiter,
                         std::string prefix, std::string postfix) {
  std::cout << prefix;
  for (int i = 0; i < vec.size(); i++) {
    std::cout << vec[i];
    if (i != vec.size() - 1) std::cout << delimiter;
  }
  std::cout << postfix;
}

std::vector<float> softmax(std::vector<float> values) {
  float max_value = *std::max_element(values.begin(), values.end());

  std::vector<float> results;

  float denom = 0.0f;
  for (auto& val : values) {
    val = expf(val - max_value);
    denom += val;
  };
  for (auto val : values) results.push_back(val / denom);

  return results;
}

template <typename T>
std::vector<size_t> argsort(const std::vector<T>& array, bool desc) {
  std::vector<size_t> indices(array.size());
  std::iota(indices.begin(), indices.end(), 0);

  std::sort(indices.begin(), indices.end(),
            [&array, desc](size_t i1, size_t i2) {
              return desc ? array[i1] > array[i2] : array[i1] < array[i2];
            });

  return indices;
}

int main(int argc, char* argv[]) {
  if (argc < 3) {
    std::cout << "Usage: " << argv[0] << " MODEL_PATH IMAGE_PATH" << std::endl;
    return 0;
  }
  std::string model_file_path = std::string(argv[1]);
  std::string image_path = std::string(argv[2]);

  std::string line;
  std::ifstream file("./resource/imagenet_class_names.txt");
  std::vector<std::string> class_names;
  if (file.is_open())
    while (std::getline(file, line)) {
      class_names.push_back(line);
    }
  else
    throw std::runtime_error("Class name file does not exist.");

  bool use_cuda = false;
  Ort::Session session{nullptr};
  Ort::SessionOptions session_options;
  Ort::AllocatorWithDefaultOptions allocator;

#ifdef CUDA_FOUND
  int device_id = 0;
  if (use_cuda) {
    Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(
        session_options, device_id));
  }
  std::cout << "CUDA Library is found. Session will be created on CUDA:"
            << device_id << std::endl;
#else
  if (use_cuda) {
    std::cout << "CUDA Library is not found. Session will be created on CPU"
              << std::endl;
    use_cuda = false;
  }
#endif

  session_options.SetGraphOptimizationLevel(
      GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "OnnxRuntime C++ Test");

  try {
    session = Ort::Session(env, model_file_path.c_str(), session_options);
  } catch (Ort::Exception& err) {
    std::cout << err.what() << std::endl;
    return -1;
  }

  Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(
      OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

  std::cout << "Ort Session is created." << std::endl;

  std::vector<TensorInfo> input_tensor_infos;

  size_t num_inputs = session.GetInputCount();
  std::vector<std::string> input_names(num_inputs, "");
  std::cout << "inputs:" << std::endl;
  for (int i = 0; i < num_inputs; i++) {
    std::string node_name = session.GetInputNameAllocated(i, allocator).get();

    auto input_type_info = session.GetInputTypeInfo(i);
    auto tensor_type_n_shape = input_type_info.GetTensorTypeAndShapeInfo();
    auto shape = tensor_type_n_shape.GetShape();
    auto type = tensor_type_n_shape.GetElementType();

    printf("[%d]\n", i);
    std::cout << "\tname : " << node_name << std::endl;
    std::cout << "\tshape: ";
    PrintVectorElements(shape);
    std::cout << std::endl;
    std::cout << "\tdtype: " << TENSOR_DTYPES[static_cast<int>(type)]
              << std::endl;

    input_names[i] = node_name;

    int num_elements = 1;
    for (auto& val : shape) num_elements *= val;
    input_tensor_infos.emplace_back((TensorInfo){
        .shape = shape, .dtype = type, .num_elements = num_elements});
  };

  std::vector<TensorInfo> output_tensor_infos;

  size_t num_outputs = session.GetOutputCount();
  std::vector<std::string> output_names(num_outputs, "");
  std::cout << "outputs:" << std::endl;
  for (int i = 0; i < num_outputs; i++) {
    std::string node_name = session.GetOutputNameAllocated(i, allocator).get();

    auto input_type_info = session.GetOutputTypeInfo(i);
    auto tensor_type_n_shape = input_type_info.GetTensorTypeAndShapeInfo();
    auto shape = tensor_type_n_shape.GetShape();
    auto type = tensor_type_n_shape.GetElementType();

    printf("[%d]\n", i);
    std::cout << "\tname : " << node_name << std::endl;
    std::cout << "\tshape: ";
    PrintVectorElements(shape);
    std::cout << std::endl;
    std::cout << "\tdtype: " << TENSOR_DTYPES[static_cast<int>(type)]
              << std::endl;

    output_names[i] = node_name;

    int num_elements = 1;
    for (auto& val : shape) num_elements *= val;
    output_tensor_infos.emplace_back((TensorInfo){
        .shape = shape, .dtype = type, .num_elements = num_elements});
  };
  std::cout << std::endl;

  cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
  std::cout << "Input Image: " << image_path << std::endl;
  std::cout << "\tshape(HxWxC): " << image.rows << "x" << image.cols << "x"
            << image.channels() << std::endl;

  // Preprocessing
  // Refer:
  // https://pytorch.org/vision/0.17/models/generated/torchvision.models.resnet50.html
  cv::Mat preprocessed;
  cv::resize(image, preprocessed, cv::Size(256, 256), cv::INTER_LINEAR);
  preprocessed = preprocessed(cv::Rect(16, 16, 224, 224));

  cv::cvtColor(preprocessed, preprocessed, cv::COLOR_BGR2RGB);
  preprocessed.convertTo(preprocessed, CV_32FC3);
  preprocessed /= 255.0f;

  cv::subtract(preprocessed, cv::Scalar(0.485, 0.456, 0.406), preprocessed);
  cv::divide(preprocessed, cv::Scalar(0.229, 0.224, 0.225), preprocessed);

  // Inference
  std::vector<const char*> input_names_char(num_inputs);
  std::transform(std::begin(input_names), std::end(input_names),
                 std::begin(input_names_char),
                 [&](const std::string& str) { return str.c_str(); });
  std::vector<const char*> output_names_char(num_outputs);
  std::transform(std::begin(output_names), std::end(output_names),
                 std::begin(output_names_char),
                 [&](const std::string& str) { return str.c_str(); });

  // cv::Mat to Tensor
  cv::dnn::blobFromImage(preprocessed, preprocessed);

  std::vector<Ort::Value> input_tensors;
  input_tensors.emplace_back(Ort::Value::CreateTensor<float>(
      mem_info, (float*)preprocessed.data, input_tensor_infos[0].num_elements,
      input_tensor_infos[0].shape.data(), 4));

  auto output_tensors =
      session.Run(Ort::RunOptions{nullptr}, input_names_char.data(),
                  input_tensors.data(), input_names_char.size(),
                  output_names_char.data(), output_names_char.size());

  std::vector<float> output_tensor(output_tensor_infos[0].num_elements);
  memcpy(output_tensor.data(),
         output_tensors.front().GetTensorMutableData<float>(),
         sizeof(float) * output_tensor_infos[0].num_elements);

  std::vector<float> probabilities = softmax(output_tensor);
  std::vector<size_t> sorted_indices = argsort(probabilities, true);

  std::cout << std::fixed << std::setprecision(3) << std::endl;
  for (int i = 0; i < 5; i++) {
    size_t index = sorted_indices[i];
    std::cout << class_names[index] << ": " << probabilities[index] * 100 << "%"
              << std::endl;
  }

  return 0;
}