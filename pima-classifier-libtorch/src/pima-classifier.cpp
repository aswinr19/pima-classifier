#include <torch/torch.h>
#include <fstream>
#include <iomanip>
#include <random>

using namespace torch::indexing;

typedef std::vector<double> dVec1D;
typedef std::vector<std::vector<double>> dVec2D;

struct Data {
    dVec2D X;
    dVec1D y;
  };

struct Data2 {
    dVec2D XTrain;
    dVec2D XTest;
    dVec1D yTrain;
    dVec1D yTest;
  };

struct TensorData {
    torch::Tensor XBatch;
    torch::Tensor yBatch;
  };

TensorData BatchSplit(torch::Tensor X, torch::Tensor y, int batchSize, int batchIndex) {

    TensorData data;

    int samples = X.sizes()[0];
    auto options1 = torch::TensorOptions().dtype(at::kDouble);

    if (batchIndex + batchSize > X.sizes()[0] - 1) {
        batchSize = X.sizes()[0] - batchIndex;
      }

    torch::Tensor tempX = X.index({Slice(batchIndex, batchIndex + batchSize)});
    torch::Tensor tempy = y.index({Slice(batchIndex, batchIndex + batchSize)});

    data.XBatch = tempX;
    data.yBatch = tempy;

    return  data;
  }


Data2 TrainTestSplit(Data data, float trainSize) {

    Data2 trainTestData;
    int samples = trainSize * data.X.size();

    trainTestData.XTrain.resize(samples, dVec1D(data.X[0].size(), 0));
    trainTestData.XTest.resize(data.X.size() - samples, dVec1D(data.X[0].size(), 0));
    trainTestData.yTrain.resize(samples, 0);
    trainTestData.yTest.resize(data.y.size() - samples, 0);

    for (int i = 0; i < samples; i++) {
            trainTestData.XTrain[i] = data.X[i];
            trainTestData.yTrain[i] = data.y[i];
      }
    for (int i = samples, idx = 0; i < data.X.size(); i++, idx++) {
        trainTestData.XTest[idx] = data.X[i];
        trainTestData.yTest[idx] = data.y[i];
      }

      return trainTestData;
  }

Data AddNoise(Data data, float mean, float stdDev, float noisySamples) {
  std::default_random_engine generator;
  std::normal_distribution<double> distribution(mean , stdDev);

  int samples = data.X.size() * noisySamples;

  dVec2D temp(samples, dVec1D(data.X[0].size(), 0));

  for (int i = 0; i < samples; i++) {
    for (int j = 0; j < data.X[0].size(); j++) {
      temp[i][j] += data.X[i][j] + distribution(generator);
    }
  }
  for (int i = 0; i < samples; i++) {
      data.X.push_back(temp[i]);
      data.y.push_back(data.y[i]);
    }

 return data;
}

Data NormalizeMinMaxData(Data data) {

    dVec1D min(data.X[0].size(), std::numeric_limits<double>::max());
    dVec1D max(data.X[0].size(), std::numeric_limits<double>::min());

   for (int i = 0; i < data.X[0].size(); ++i) {
      for (int j = 0; j < data.X.size(); ++j) {
         if (data.X[j][i] < min[i])  min[i] = data.X[j][i];
         if (data.X[j][i] > max[i])  max[i] = data.X[j][i];
        }
    }

   for (int i = 0; i < data.X[0].size(); ++i) {
      for (int j = 0; j < data.X.size(); ++j) {
          data.X[j][i] = (data.X[j][i] - min[i]) / (max[i] - min[i]);
        }
    }

    return data;
  }

Data SplitFeaturesAndClasses(dVec2D rowData, int samples, int features, int splitIndex) {

 Data data;
  data.X.resize(samples, dVec1D(features - 1, 0.0));
  data.y.resize(samples);

  for (int i = 0; i < samples; i++) {
    for (int j = 0; j < splitIndex; j++) {
       data.X[i][j] = rowData[i][j];
      }
      data.y[i] = rowData[i][splitIndex];
    }
    return data;
}

dVec2D ReadDataset(const std::string& path) {
  std::ifstream datasetFile(path);

  if (!datasetFile.is_open())
    throw std::runtime_error("Could not open shape file!\n");

    dVec2D dataset;

    std::string line;

    while(std::getline(datasetFile, line)){
        std::istringstream ss(line);

        std::string field;
        dVec1D row;

        while(std::getline(ss, field, ',')) {
            row.push_back(stod(field));
          }
          dataset.push_back(row);
      }
      return dataset;
  }

struct Model: torch::nn::Module {
    Model() {
      fc1 = register_module("fc1", torch::nn::Linear(8, 10));
      fc2 = register_module("fc2", torch::nn::Linear(10, 8));
      fc3 = register_module("fc3", torch::nn::Linear(8, 1));
      }

      torch::Tensor forward(torch::Tensor x) {
          x = torch::relu(fc1->forward(x));
          x = torch::relu(fc2->forward(x));
          x = torch::sigmoid(fc3->forward(x));

          return x;
        }

      torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
  };

int main() {

  std::string path = "../data/pima.csv";
  dVec2D dataset = ReadDataset(path);
  Data data = SplitFeaturesAndClasses(dataset, dataset.size(), dataset[0].size(), dataset[0].size() -1);
  int batchSize = 12;
  int epoch = 1000;

  data = NormalizeMinMaxData(data);

  data = AddNoise(data, 0, 0.1, 0.2);

  Data2 trainTestData = TrainTestSplit(data, 0.8);

  Data data2;
  data2.X = trainTestData.XTrain;
  data2.y = trainTestData.yTrain;

  Data2 trainValidateData = TrainTestSplit(data2, 0.8);
  
  auto options1 = torch::TensorOptions().dtype(at::kDouble);

  auto tensorXTrain = torch::zeros({trainTestData.XTrain.size(), trainTestData.XTrain[0].size()}, options1);
  auto tensorYTrain = torch::zeros({ trainTestData.yTrain.size(), 1}, options1);

  auto tensorXValidate = torch::zeros({trainValidateData.XTest.size(), trainValidateData.XTest[0].size()}, options1);
  auto tensorYValidate = torch::zeros({ trainValidateData.yTest.size(), 1}, options1);

  auto tensorXTest = torch::zeros({trainTestData.XTest.size(), trainTestData.XTest[0].size()}, options1);
  auto tensorYTest = torch::zeros({ trainTestData.yTest.size(), 1}, options1);


    for (int i = 0; i < trainTestData.XTrain.size(); ++i) {
        auto temp = torch::from_blob(trainTestData.XTrain[i].data(), {trainTestData.XTrain[0].size()}, options1);
        tensorXTrain[i] = temp;
        tensorYTrain[i] = trainTestData.yTrain[i];
      }

  for (int i = 0; i < trainValidateData.XTest.size(); ++i) {
        auto temp = torch::from_blob(trainValidateData.XTest[i].data(), {trainValidateData.XTest[0].size()}, options1);
        tensorXValidate[i] = temp;
        tensorYValidate[i] = trainValidateData.yTest[i];
      }


    for (int i = 0; i < trainTestData.XTest.size(); ++i) {
        auto temp = torch::from_blob(trainTestData.XTest[i].data(), {trainTestData.XTest[0].size()}, options1);
        tensorXTest[i] = temp;
        tensorYTest[i] = trainTestData.yTest[i];
      }

      auto nn = std::make_shared<Model>();

      //torch::optim::SGD optimizer(nn->parameters(), 0.0001);
      torch::optim::Adam optimizer(nn->parameters(), 0.0001);
      torch::nn::BCELoss loss_fn;

      nn->to(torch::kDouble);

      for(int i = 0; i < epoch; ++i) {
          nn->train();
          optimizer.zero_grad();

          double lossEpoch = 0;

        for (int j = 0, count = 1; j < tensorXTrain.sizes()[0]; j += batchSize, count++) {
          TensorData batchData = BatchSplit(tensorXTrain, tensorYTrain, batchSize, j);

          torch::Tensor prediction = nn->forward(batchData.XBatch.to(torch::kDouble));
          auto loss = loss_fn(prediction, batchData.yBatch.to(torch::kDouble));
          loss.backward();
          optimizer.step();
          lossEpoch += loss.item<double>();
        }
          
        double trainLossEpoch = lossEpoch / double(tensorXTrain.sizes()[0] / batchSize);
        
        TensorData batchData = BatchSplit(tensorXValidate, tensorYValidate, tensorXValidate.sizes()[0], 0);
        torch::Tensor predictionVal = nn->forward(batchData.XBatch.to(torch::kDouble));
        auto valLoss = loss_fn(predictionVal, batchData.yBatch.to(torch::kDouble));

       std::cout << "Epoch: " << i + 1 << " , Train Loss: " << trainLossEpoch <<" , Validation Loss: " << valLoss.item() << std::endl; 
      }

      torch::Tensor prediction = nn->forward(tensorXTest);

      int count = 0;
      for (int i = 0; i < prediction.sizes()[0]; i++) {
           if (torch::equal(torch::round(prediction[i]), tensorYTest[i])){
              count++;
            }
        }
        std::cout <<"test accuracy: " << std::setprecision(10) << double( count ) / double (prediction.sizes()[0] ) * 100 << std::endl;

    return 0;
  }
