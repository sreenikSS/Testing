#include <iostream>
#include <fstream>
#include <string>
#include <iterator>
#include <queue>
#include <mlpack/core.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/init_rules/glorot_init.hpp>
#include <mlpack/methods/ann/init_rules/const_init.hpp>
#include <mlpack/methods/ann/init_rules/he_init.hpp>
#include <mlpack/methods/ann/init_rules/kathirvalavakumar_subavathi_init.hpp>
#include <mlpack/methods/ann/init_rules/lecun_normal_init.hpp>
#include <mlpack/methods/ann/init_rules/nguyen_widrow_init.hpp>
#include <mlpack/methods/ann/init_rules/oivs_init.hpp>
#include <mlpack/methods/ann/init_rules/orthogonal_init.hpp>
#include <mlpack/methods/ann/init_rules/random_init.hpp>
#include <mlpack/methods/ann/loss_functions/cross_entropy_error.hpp>
//#include <mlpack/methods/ann/loss_functions/earth_mover_distance.hpp>
#include <mlpack/methods/ann/loss_functions/kl_divergence.hpp>
#include <mlpack/methods/ann/loss_functions/mean_squared_error.hpp>
#include <mlpack/methods/ann/loss_functions/sigmoid_cross_entropy_error.hpp>
//#include <mlpack/methods/ann/loss_functions/reconstruction_loss.hpp>
#include <mlpack/methods/ann/activation_functions/softsign_function.hpp>
#include <mlpack/methods/ann/activation_functions/swish_function.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/layer/base_layer.hpp>
#include <ensmallen.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/algorithm/string/trim.hpp>
#include <boost/archive/xml_iarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/tokenizer.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/foreach.hpp>
#include <boost/algorithm/string.hpp>

using namespace mlpack;
using namespace mlpack::ann;
using namespace mlpack::optimization;
//using namespace ens;
using namespace arma;
using namespace std;
using namespace boost::property_tree;

class Dataset
{
 private:
   arma::mat trainX, trainY, validX, validY;
 public:
   Dataset(){}
   Dataset(arma::mat& trainX, arma::mat& trainY)
   {
     this->trainX = trainX;
     this->trainY = trainY;
   }
   Dataset(arma::mat& trainX, arma::mat& trainY,
           arma::mat& validX, arma::mat& validY)
   {
     this->trainX = trainX;
     this->trainY = trainY;
     this->validX = validX;
     this->validX = validY;
   }
   void setTrainSet(arma::mat& trainX, arma::mat& trainY)
   {
     this->trainX = trainX;
     this->trainY = trainY;
   }
   void setValidSet(arma::mat& validX, arma::mat& validY)
   {
     this->validX = validX;
     this->validY = validY;
   }
   arma::mat getTrainX()
   {
     return trainX;
   }
   arma::mat getTrainY()
   {
     return trainY;
   }
   arma::mat getValidX()
   {
     return validX;
   }
   arma::mat getValidY()
   {
     return validY;
   }
};

void printError()
{
  cout << "Error" << endl;
  exit(1);
}

void printMap(map<string, double> params)
{
  map<string, double>::iterator itr;
  cout << "Map details:" << endl;
  for (itr = params.begin(); itr != params.end(); ++itr)
  {
    cout << itr->first << " : " << itr->second << endl;
  }
}

void updateParams(map<string, double> &origParams, map<string, double> &newParams)
{
  map<string, double>::iterator itr;
  for (itr = newParams.begin(); itr != newParams.end(); ++itr)
  {
    map<string, double>::iterator itr2 = origParams.find(itr->first);
    itr2->second = newParams.at(itr->first);
  }
  printMap(origParams);
}

template <typename OptimizerType, typename LossType, typename InitType>
void trainModel(OptimizerType optimizer, FFN<LossType, InitType>& model,
                int cycles, Dataset& dataset)
{
  const arma::mat trainX = dataset.getTrainX();
  const arma::mat trainY = dataset.getTrainY();
  const arma::mat validX = dataset.getValidX();
  const arma::mat validY = dataset.getValidY();
  for(int i = 1; i <= cycles; i++)
  {
    model.Train(trainX, trainY, optimizer);
    arma::mat predOut;
    model.Predict(trainX, predOut);
    predOut.transform([](double val) { return roundf(val);});
    double trainAccuracy = 0;
    for (int j=0; j < trainY.n_cols; j++)
    {
        trainAccuracy += ( (int) predOut[j] == (int) trainY[j]);
    }
    trainAccuracy /= (double) trainY.n_cols;
    model.Predict(validX, predOut);
    predOut.transform([](double val) { return roundf(val);});
    double validAccuracy = 0;
    for (int j=0; j < validY.n_cols; j++)
    {
        validAccuracy += ( (int) predOut[j] == (int) validY[j]);
    }
    validAccuracy /= (double) validY.n_cols;
    cout << "Cycle: " << i << " Training accuracy: " << trainAccuracy <<
        " Validation accuracy: " << validAccuracy << endl;
  }
}

template <typename LossType, typename InitType>
void createModel(string optimizerType,
                 map<string, double>& optimizerParams,
                 queue<LayerTypes<> >& layers,
                 const Dataset& dataset)
{
  FFN<LossType, InitType> model;
  while (!layers.empty())
  {
    model.Add(layers.front());
    layers.pop();
  }
  map<string, double> origParams;
  origParams["cycles"] = 1;
  string optimizerGroup1[] = {"adadelta", "adagrad", "adam",
      "adamax", "amsgrad", "bigbatchsgd", "momentumsgd",
      "nadam", "nadamax", "nesterovmomentumsgd",
      "optimisticadam", "rmsprop", "sarah", "sgd", "sgdr",
      "snapshotsgdr", "smorms3", "svrg", "spalerasgd"};

  for (string& itr : optimizerGroup1)
  {
    if (itr == optimizerType)
    {
      origParams["stepsize"] = 0.01;
      origParams["batchsize"] = 32;
      origParams["maxiterations"] = 100000;
      origParams["tolerance"] = 1e-5;
      origParams["shuffle"] = true;
      break;
    }
  }

  if (optimizerType == "adadelta")
  {
    origParams["stepsize"] = 1.0;
    origParams["rho"] = 0.95;
    origParams["epsilon"] = 1e-6;
    origParams["resetpolicy"] = true;
    updateParams(origParams, optimizerParams);
    ens::AdaDelta optimizer(origParams["stepsize"], origParams["batchsize"],
        origParams["rho"], origParams["epsilon"], origParams["maxiterations"],
        origParams["tolerance"], origParams["shuffle"],
        origParams["resetpolicy"]);
    //trainModel<ens::AdaDelta, LossType, InitType>(optimizer, model, origParams["cycles"], dataset);
  }
  else if (optimizerType == "adagrad")
  {
    origParams["epsilon"] = 1e-8;
    origParams["resetpolicy"] = true;
    updateParams(origParams, optimizerParams);
    ens::AdaGrad optimizer(origParams["stepsize"], origParams["batchsize"],
        origParams["epsilon"], origParams["maxiterations"],
        origParams["tolerance"], origParams["shuffle"],
        origParams["resetpolicy"]);
    //trainModel<ens::AdaGrad, LossType, InitType>(optimizer, model, origParams["cycles"], dataset);
  }
  else if (optimizerType == "adam" || optimizerType == "adamax" || 
      optimizerType == "amsgrad" || optimizerType == "optimisticadam" ||
      optimizerType == "nadamax" || optimizerType == "nadam")
  {
    origParams["stepsize"] = 0.001;
    origParams["beta1"] = 0.9;
    origParams["beta2"] = 0.999;
    origParams["epsilon"] = 1e-8; // eps
    origParams["resetpolicy"] = true;

    updateParams(origParams, optimizerParams);
    if (optimizerType == "adam")
    {
      ens::Adam optimizer(origParams["stepsize"], origParams["batchsize"],
          origParams["beta1"], origParams["beta2"], origParams["epsilon"],
          origParams["maxiterations"], origParams["tolerance"],
          origParams["shuffle"], origParams["resetpolicy"]);
      //trainModel<ens::Adam, LossType, InitType>(optimizer, model, origParams["cycles"], dataset);
    }
    else if (optimizerType == "adamax")
    {
      ens::AdaMax optimizer(origParams["stepsize"], origParams["batchsize"],
          origParams["beta1"], origParams["beta2"], origParams["epsilon"],
          origParams["maxiterations"], origParams["tolerance"],
          origParams["shuffle"], origParams["resetpolicy"]);
      //trainModel<ens::AdaMax, LossType, InitType>(optimizer, model, origParams["cycles"], dataset);
    }
    else if (optimizerType == "amsgrad")
    {
      ens::AMSGrad optimizer(origParams["stepsize"], origParams["batchsize"],
          origParams["beta1"], origParams["beta2"], origParams["epsilon"],
          origParams["maxiterations"], origParams["tolerance"],
          origParams["shuffle"], origParams["resetpolicy"]);
      //trainModel<ens::AMSGrad, LossType, InitType>(optimizer, model, origParams["cycles"], dataset);
    }
    else if (optimizerType == "optimisticadam")
    {
      ens::OptimisticAdam optimizer(origParams["stepsize"],
          origParams["batchsize"], origParams["beta1"],
          origParams["beta2"], origParams["epsilon"],
          origParams["maxiterations"], origParams["tolerance"],
          origParams["shuffle"], origParams["resetpolicy"]);
      //trainModel<ens::OptimisticAdam>(optimizer, origParams["cycles"],
      //    dataset);
    }
    else if (optimizerType == "nadamax")
    {
      ens::NadaMax optimizer(origParams["stepsize"], origParams["batchsize"],
          origParams["beta1"], origParams["beta2"], origParams["epsilon"],
          origParams["maxiterations"], origParams["tolerance"],
          origParams["shuffle"], origParams["resetpolicy"]);
      //trainModel<ens::NadaMax, LossType, InitType>(optimizer, model, origParams["cycles"], dataset);
    }
    else if (optimizerType == "nadam")
    {
      ens::Nadam optimizer(origParams["stepsize"], origParams["batchsize"],
          origParams["beta1"], origParams["beta2"], origParams["epsilon"],
          origParams["maxiterations"], origParams["tolerance"],
          origParams["shuffle"], origParams["resetpolicy"]);
      //trainModel<ens::Nadam, LossType, InitType>(optimizer, model, origParams["cycles"], dataset);
    }
  }
  else if (optimizerType == "iqn")
  {
    origParams["stepsize"] = 0.01;
    origParams["batchsize"] = 10;
    origParams["maxiterations"] = 100000;
    origParams["tolerance"] = 1e-5;
    updateParams(origParams, optimizerParams);
    ens::IQN optimizer(origParams["stepsize"], origParams["batchsize"],
        origParams["maxiterations"], origParams["tolerance"]);
    //trainModel<ens::IQN, LossType, InitType>(optimizer, model, origParams["cycles"], dataset);
  }
  else if (optimizerType == "katyusha")
  {
    origParams["convexity"] = 1.0;
    origParams["lipschitz"] = 10.0;
    origParams["batchsize"] = 10;
    origParams["maxiterations"] = 100000;
    origParams["inneriterations"] = 0;
    origParams["tolerance"] = 1e-5;
    origParams["shuffle"] = true;
    updateParams(origParams, optimizerParams);
    ens::Katyusha optimizer(origParams["stepsize"], origParams["batchsize"],
        origParams["maxiterations"], origParams["tolerance"]);
    //trainModel<ens::Katyusha, LossType, InitType>(optimizer, model, origParams["cycles"], dataset);
  }
  else if (optimizerType == "momentumsgd")
  {
    updateParams(origParams, optimizerParams);
    // The MomentumUpdate() parameter can be made modifiable
    ens::MomentumSGD optimizer(origParams["stepsize"],
        origParams["batchsize"], origParams["maxiterations"],
        origParams["tolerance"], origParams["shuffle"]);
    //trainModel<ens::MomentumSGD, LossType, InitType>(optimizer, model, origParams["cycles"], dataset);
  }
  else if (optimizerType == "nesterovmomentumsgd")
  {
    updateParams(origParams, optimizerParams);
    // The MomentumUpdate() parameter can be made modifiable
    ens::NesterovMomentumSGD optimizer(origParams["stepsize"],
        origParams["batchsize"], origParams["maxiterations"],
        origParams["tolerance"], origParams["shuffle"]);
    //trainModel<ens::NesterovMomentumSGD>(optimizer, origParams["cycles"],
    //    dataset);
  }
  else if (optimizerType == "rmsprop")
  {
    origParams["alpha"] = 0.99;
    origParams["epsilon"] = 1e-8;
    origParams["resetpolicy"] = true;
    updateParams(origParams, optimizerParams);
    // tolerance set to 1e-5
    ens::RMSProp optimizer(origParams["stepsize"], origParams["batchsize"],
        origParams["alpha"], origParams["epsilon"],
        origParams["maxiterations"], origParams["tolerance"],
        origParams["shuffle"], origParams["resetpolicy"]);
    //trainModel<ens::RMSProp, LossType, InitType>(optimizer, model, origParams["cycles"], dataset);
  }
  else if (optimizerType == "sarah")
  {
    origParams["inneriterations"] = 0;
    updateParams(origParams, optimizerParams);
    ens::SARAH optimizer(origParams["stepsize"], origParams["batchsize"],
        origParams["maxiterations"], origParams["inneriterations"],
        origParams["tolerance"], origParams["shuffle"]);
    //trainModel<ens::SARAH, LossType, InitType>(optimizer, model, origParams["cycles"], dataset);
  }
  else if (optimizerType == "sgd")
  {
    updateParams(origParams, optimizerParams);
    ens::StandardSGD optimizer(origParams["stepsize"], origParams["batchsize"],
        origParams["maxiterations"], origParams["tolerance"],
        origParams["shuffle"]);
    //trainModel<ens::StandardSGD, LossType, InitType>(optimizer, model, origParams["cycles"], dataset);
  }
  else if (optimizerType == "sgdr")
  {
    origParams["epochrestart"] = 50;
    origParams["multfactor"] = 2.0;
    origParams["batchsize"] = 1000;
    origParams["resetpolicy"] = true;
    updateParams(origParams, optimizerParams);
    ens::SGDR<> optimizer(origParams["epochrestart"], origParams["multfactor"],
        origParams["batchsize"], origParams["stepsize"], 
        origParams["maxiterations"], origParams["tolerance"],
        origParams["shuffle"], ens::MomentumUpdate(0.5),
        origParams["resetpolicy"]);
    //trainModel<ens::SGDR<> , LossType, InitType>(optimizer, model, origParams["cycles"], dataset);
  }
  else if (optimizerType == "snapshotsgdr")
  {
    origParams["epochrestart"] = 50;
    origParams["multfactor"] = 2.0;
    origParams["batchsize"] = 1000;
    origParams["snapshots"] = 5;
    origParams["accumulate"] = true;
    origParams["resetpolicy"] = true;
    updateParams(origParams, optimizerParams);
    ens::SnapshotSGDR<> optimizer(origParams["epochrestart"],
        origParams["multfactor"], origParams["batchsize"],
        origParams["stepsize"], origParams["maxiterations"],
        origParams["tolerance"], origParams["shuffle"],
        origParams["snapshots"], origParams["accumulate"],
        ens::MomentumUpdate(0.5), origParams["resetpolicy"]);
    //trainModel<ens::SnapshotSGDR<> , LossType, InitType>(optimizer, model, origParams["cycles"], dataset);
  }
  else if (optimizerType == "smorms3")
  {
    origParams["stepsize"] = 0.001;
    origParams["epsilon"] = 1e-16;
    origParams["resetpolicy"] = true;
    updateParams(origParams, optimizerParams);
    ens::SMORMS3 optimizer(origParams["stepsize"], origParams["batchsize"], 
        origParams["epsilon"], origParams["maxiterations"],
        origParams["tolerance"], origParams["shuffle"],
        origParams["resetpolicy"]);
    //trainModel<ens::SMORMS3, LossType, InitType>(optimizer, model, origParams["cycles"], dataset);
  }
  else if (optimizerType == "svrg")
  {
    origParams["inneriterations"] = 0;
    origParams["resetpolicy"] = true;
    updateParams(origParams, optimizerParams);
    ens::SVRG optimizer(origParams["stepsize"], origParams["batchsize"],
        origParams["maxiterations"], origParams["inneriterations"],
        origParams["tolerance"], origParams["shuffle"],
        ens::SVRGUpdate(), ens::NoDecay(), origParams["resetpolicy"]);
    //trainModel<ens::SVRG, LossType, InitType>(optimizer, model, origParams["cycles"], dataset);
  }
  else if (optimizerType == "spalerasgd")
  {
    origParams["lambda"] = 0.01;
    origParams["alpha"] = 0.001;
    origParams["epsilon"] = 1e-6;
    origParams["adaptrate"] = 3.1e-8;
    origParams["resetpolicy"] = true;
    updateParams(origParams, optimizerParams);
    ens::SPALeRASGD<> optimizer(origParams["stepsize"],
        origParams["batchsize"], origParams["maxiterations"],
        origParams["tolerance"], origParams["lambda"],
        origParams["alpha"], origParams["epsilon"],
        origParams["adaptrate"], origParams["shuffle"],
        ens::NoDecay(), origParams["resetpolicy"]);
    //trainModel<ens::SPALeRASGD<> , LossType, InitType>(optimizer, model, origParams["cycles"], dataset);
  }
}

template <typename InitType>
void getLossType(string lossType, string optimizerType,
                 map<string, double> optimizerParams,
                 queue<LayerTypes<> >& layers,
                 const Dataset& dataset)
{
  if (lossType == "crossentropyerror")
  {
    createModel<CrossEntropyError<>, InitType>(optimizerType,
                                               optimizerParams,
                                               layers,
                                               dataset);
  }
  else if (lossType == "earthmoverdistance")
  {
    //createModel(EarthMoverDistance<>, InitType>();
  }
  else if (lossType == "kldivergence")
  {
    createModel<KLDivergence<>, InitType>(optimizerType,
                                          optimizerParams,
                                          layers,
                                          dataset);
  }
  else if (lossType == "meansquarederror")
  {
    createModel<MeanSquaredError<>, InitType>(optimizerType,
                                              optimizerParams,
                                              layers,
                                              dataset);
  }
  else if (lossType == "negativeloglikelihood")
  {
    createModel<NegativeLogLikelihood<>, InitType>(optimizerType,
                                                   optimizerParams,
                                                   layers,
                                                   dataset);
  }
  else if (lossType == "reconstructionloss")
  {
    //createModel<ReconstructionLoss<>, InitType>(optimizerType,
    //                                            optimizerParams);
  }
  else if (lossType == "sigmoidcrossentropyerror")
  {
    // createModel<SigmoidCrossEntropyError<>, InitType>(optimizerType,
    //                                                   optimizerParams,
    //                                                   layers,
    //                                                   trainX,
    //                                                   trainY);
  }
  else
  {
    printError();
  }
}

void getInitType(string initType, string lossType,
                 map<string, double> initDetails,
                 string optimizerType, map<string,
                 double> optimizerParams,
                 queue<LayerTypes<> >& layers,
                 const Dataset& dataset)
{
  if (initType == "const")
  {
    //initLayer = new ConstInitialization(3.0);
    map<string, double> origParams;
    origParams["initval"];
    //getLossType<ConstInitialization>(lossType);
  }
  else if (initType == "gaussian")
  {
    getLossType<GaussianInitialization>(lossType, optimizerType,
                                        optimizerParams, layers,
                                        dataset);
  }
  else if (initType == "glorot")
  {
    getLossType<GlorotInitialization>(lossType, optimizerType,
                                      optimizerParams, layers,
                                      dataset);
  }
  else if (initType == "he")
  {
    getLossType<HeInitialization>(lossType, optimizerType,
                                  optimizerParams, layers,
                                  dataset);
  }
  else if (initType == "kathirvalavakumar_subavathi")
  {
    //getLossType<KathirvalavakumarSubavathiInitialization>(lossType,
    //                                                      optimizerType,
    //                                                      optimizerParams);
  }
  else if (initType == "lecun_normal")
  {
    getLossType<LecunNormalInitialization>(lossType, optimizerType,
                                           optimizerParams, layers,
                                           dataset);
  }
  else if (initType == "nguyen_widrow")
  {
    getLossType<NguyenWidrowInitialization>(lossType, optimizerType,
                                            optimizerParams, layers,
                                            dataset);
  }
  else if (initType == "oivs")
  {
    //getLossType<OivsInitialization>(lossType, optimizerType, optimizerParams);
  }
  else if (initType == "orthogonal")
  {
    getLossType<OrthogonalInitialization>(lossType, optimizerType,
                                          optimizerParams, layers,
                                          dataset);
  }
  else if (initType == "random")
  {
    getLossType<RandomInitialization>(lossType, optimizerType,
                                      optimizerParams, layers,
                                      dataset);
  }
  else
  {
    printError();
  }
}

void testMaps()
{
  map<string, double> param1;
  map<string, double> param2;
  param1["id1"] = 1.0;
  param1["id2"] = 2.0;
  param1["id3"] = 3.0;
  param1["id4"] = 4.0;

  param2["id1"] = 11.0;
  param2["id2"] = 12.0;
  param2["id6"] = 13.0;
  param2["id7"] = 14.0;

  updateParams(param1, param2);
  printMap(param1);
}

LayerTypes<> getNetworkReference(string type, map<string, double>& newParams)
{
  map<string, double> origParams;
  LayerTypes<> layer;

  if (type == "alphadropout")
  {
    origParams["ratio"] = 0.5;
    // alphadash is the default value of -alpha*lambda
    origParams["alphadash"] = -1.758099340847376624;
    updateParams(origParams, newParams);
    layer = new AlphaDropout<>(origParams["ratio"], origParams["alphadash"]);
  }
  else if (type == "batchnorm")
  {
    layer = new BatchNorm<>();
  }
  else if (type == "constant")
  {
    origParams["outsize"];
    origParams["scalar"] = 0.0;
    layer = new Constant<>(origParams["outsize"], origParams["scalar"]);
  }
  else if (type == "convolution")
  {
    // if (len(val))
    origParams["insize"];
    origParams["outsize"];
    origParams["kw"];
    origParams["kh"];
    origParams["dw"] = 1;
    origParams["dh"] = 1;
    origParams["padw"] = 0;
    origParams["padh"] = 0;
    origParams["inputwidth"] = 0;
    origParams["inputheight"] = 0;
    updateParams(origParams, newParams);
    layer = new Convolution<>(origParams["insize"], origParams["outsize"],
        origParams["kw"], origParams["kh"], origParams["dw"], 
        origParams["dh"], origParams["padw"], origParams["padh"],
        origParams["inputwidth"], origParams["inputheight"]);
  }
  else if (type == "dropconnect")
  {
    // origParams["insize"];
    // origParams["outsize"];
    // origParams["ratio"] = 0.5;
    // updateParams(origParams, newParams);
    // layer = new DropConnect<>();
  }
  else if (type == "dropout")
  {
    origParams["ratio"] = 0.5;
    updateParams(origParams, newParams);
    layer = new Dropout<>(origParams["ratio"]);
  }
  else if (type == "fastlstm")
  {
    //origParams = {{""}}
  }
  else if (type == "gru")
  {
  }
  else if (type == "layernorm")
  {
    layer = new LayerNorm<>();
  }
  else if (type == "linearnobias")
  {
    origParams["insize"];
    origParams["outsize"];
    updateParams(origParams, newParams);
    layer = new LinearNoBias<>(origParams["insize"], origParams["outsize"]);
  }
  else if (type == "linear")
  {
    origParams["insize"];
    origParams["outsize"];
    updateParams(origParams, newParams);
    layer = new Linear<>(origParams["insize"], origParams["outsize"]);
  }
  else if (type == "maxpooling")
  {
    origParams["kw"];
    origParams["kh"];
    origParams["dw"] = 1;
    origParams["dh"] = 1;
    origParams["floor"] = 1; // 1 for true, 0 for false
    updateParams(origParams, newParams);
    layer = new MaxPooling<>(origParams["kw"], origParams["kh"],
        origParams["dw"], origParams["dh"], origParams["floor"]);
  }
  else if (type == "meanpooling")
  {
    origParams["kw"];
    origParams["kh"];
    origParams["dw"] = 1;
    origParams["dh"] = 1;
    origParams["floor"] = true;
    updateParams(origParams, newParams);
    layer = new MeanPooling<>(origParams["kw"], origParams["kh"],
        origParams["dw"], origParams["dh"], origParams["floor"]);
  }
  else if (type == "multiplyconstant")
  {
    origParams["scalar"] = 1.0;
    updateParams(origParams, newParams);
    layer = new MultiplyConstant<>(origParams["scalar"]);
  }
  else if (type == "recurrent")
  {

  }
  else if (type == "transposedconvolution")
  {
    origParams["insize"];
    origParams["outsize"];
    origParams["kw"];
    origParams["kh"];
    origParams["dw"] = 1;
    origParams["dh"] = 1;
    origParams["padw"] = 0;
    origParams["padh"] = 0;
    origParams["inputwidth"] = 0;
    origParams["inputheight"] = 0;
    updateParams(origParams, newParams);
    layer = new TransposedConvolution<>(origParams["insize"],
        origParams["outsize"], origParams["kw"],origParams["kh"],
        origParams["dw"], origParams["dh"], origParams["padw"],
        origParams["padh"], origParams["inputwidth"],
        origParams["inputheight"]);
  }
  else if (type == "identity")
  {
    layer = new IdentityLayer<>();
  }
  else if (type == "logistic")
  {
    //layer = LogisticFunction();
  }
  else if (type == "rectifier" || type == "relu")
  {
    layer = new ReLULayer<>();
  }
  else if (type == "softplus")
  {

  }
  else if (type == "softsign")
  {
    //layer = SoftsignFunction;
  }
  else if (type == "swish")
  {
    //layer = new Swish
  }
  else if (type == "tanh")
  {
    layer = new TanHLayer<>();
  }
  else if (type == "elu")
  {
    origParams["alpha"];
    updateParams(origParams, newParams);
    layer = new ELU<>(origParams["alpha"]);
  }
  else if (type == "selu")
  {
    layer = new SELU();
  }
  else if (type == "hardtanh")
  {
    origParams["maxvalue"] = 1;
    origParams["minvalue"] = -1;
    updateParams(origParams, newParams);
    layer = new HardTanH<>(origParams["maxvalue"], origParams["minvalue"]);
  }
  else if (type == "leakyrelu")
  {
    origParams["alpha"] = 0.03;
    updateParams(origParams, newParams);
    layer = new LeakyReLU<>(origParams["alpha"]);
  }
  else if (type == "prelu")
  {
    origParams["alpha"] = 0.03; // userAlpha
    layer = new PReLU<>(origParams["alpha"]);
  }
  else if (type == "sigmoid")
  {
    layer = new SigmoidLayer<>();
  }
  else if (type == "softmax")
  {
    layer = new LogSoftMax<>();
  }
  else
  {
    printError();
  }
  return layer;
}

void traverseModel(const ptree& tree, const Dataset& dataset, double& inSize)
{
  const ptree &loss = tree.get_child("loss");
  const ptree &init = tree.get_child("init");
  const ptree &optimizer = tree.get_child("optimizer");
  const ptree &network = tree.get_child("network");
  queue<LayerTypes<> > layers;

  string lossType = loss.get_value<string>();

  map<string, double> initDetails;
  string initType;
  BOOST_FOREACH (ptree::value_type const &v, init.get_child(""))
  {
    const ptree &attributes = v.second;
    if (v.first == "type")
    {
      initType = attributes.get_value<string>();
    }
    else
    {
      initDetails[v.first] = attributes.get_value<double>();
    }
    //cout << attributes.get_value<string>() << endl;
  }
  cout << "type : " << initType << endl;
  printMap(initDetails);

  map<string, double> optimizerDetails;
  string optimizerType;
  BOOST_FOREACH (ptree::value_type const &v, optimizer.get_child(""))
  {
    const ptree &attributes = v.second;
    if (v.first == "type")
    {
      optimizerType = attributes.get_value<string>();
    }
    else
    {
      optimizerDetails[v.first] = attributes.get_value<double>();
    }
  }
  cout << "type : " << optimizerType << endl;
  printMap(optimizerDetails);

  BOOST_FOREACH (ptree::value_type const &v, network.get_child(""))
  {
    const ptree &layerWhole = v.second;
    map<string, double> params;
    string type;
    BOOST_FOREACH (ptree::value_type const &v2, layerWhole.get_child(""))
    {
      const ptree &layerInner = v2.second;
      //cout << v2.first << "\t";
      //cout << layerInner.get_value<string>() << endl;
      string key = boost::erase_all_copy(v2.first, "_");
      boost::to_lower(key);
      if (key == "type")
      {
        type = layerInner.get_value<string>();
      }
      else if (key == "units")
      {
        params["insize"] = inSize;
        inSize = params["outsize"] = layerInner.get_value<double>();
      }
      else
      {
        params[key] = layerInner.get_value<double>();
      }
    }
    printMap(params);
    layers.push(getNetworkReference(type, params));
    cout << endl;
  }
  getInitType(initType, lossType, initDetails, optimizerType,
      optimizerDetails, layers, dataset);
}

boost::property_tree::ptree loadProperties(string fileName, const Dataset& dataset,
                                           double inSize)
{
  ptree pt;
  read_json("network2.json", pt);
  ptree::const_iterator end = pt.end();
  traverseModel(pt, dataset, inSize);
  return pt;
}
#include <mlpack/core/data/split_data.hpp>
int main()
{
  //arma::mat trainX, trainY;
  string fileName;
  arma::mat dataset2;
  data::Load("train_set_ap_1.csv", dataset2, true);
  cout << "Data loaded" << endl;
  arma::mat train, valid;
  data::Split(dataset2, train, valid, 0.3);
  arma::mat trainX = normalise(train.submat(0, 0, train.n_rows-2, train.n_cols-1));
  arma::mat trainY = train.row(train.n_rows-1);
  arma::mat validX = normalise(valid.submat(0, 0, valid.n_rows-2, valid.n_cols-1));
  arma::mat validY = valid.row(valid.n_rows-1);
  const Dataset dataset(trainX, trainY, validX, validY);
  loadProperties(fileName, dataset, trainX.n_rows);
  //testMaps();
  return 0;
}
