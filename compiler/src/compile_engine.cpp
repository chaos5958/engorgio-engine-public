#include <iostream>
#include <chrono>
#include <set>
#include <filesystem>
#include "compile_engine.h"
#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "logger.h"
#include "parserOnnxConfig.h"
#include "NvInfer.h"
#include "date/date.h"


using namespace std;
namespace fs = std::filesystem;


CompileEngine::CompileEngine(CompileEngineProfile &cprofile)
{
    num_threads_ = cprofile.num_threads;
    log_dir_ = cprofile.log_dir;
    save_log_ = cprofile.save_log;

    // launch threads
    preopt_worker_ = new std::thread([this](){this->PreOptHandler();});
    for (int i = 0; i < num_threads_; i++)
    {
        opt_workers_.push_back(new std::thread([this, i](){this->OptHandler(i);}));
    }
}

CompileEngine::~CompileEngine()
{
    for (auto log : logs_)
    {
        delete log;
    }

    // stop threads
    OptEvent event;
    event.type = OptType::kJoin;
    preopt_emutex_.lock();
    preopt_events_.push_back(event);
    preopt_emutex_.unlock();
    preopt_worker_->join();

    opt_emutex_.lock();
    // for (auto worker : opt_workers_)
    for (std::size_t i = 0; i < opt_workers_.size(); i++)
    {
        opt_events_.push_back(event);
    }
    opt_emutex_.unlock();
    for (auto worker : opt_workers_)
    {
        worker->join();
    }

    // prelease in-memory dnns
    for (auto & [key, val] : preopt_dnns_)
    {
        val->destroy();
    }
    while(opt_dnns_.size() > 0)
    {
        opt_dnns_.front()->opt_dnn->destroy();
        opt_dnns_.pop_front();
    }
}

bool CompileEngine::PreOptInternal(OptEvent &event)
{
    OnnxModel *onnx_model = event.onnx_model;

    auto start = std::chrono::high_resolution_clock::now();

    if (!onnx_model->is_loaded_)
    {
        std::cerr << "onnx model is not loaded" << std::endl;
        return false;
    }
    preopt_dmutex.lock_shared();
    if (preopt_dnns_.find(onnx_model->name_) != preopt_dnns_.end())
    {
        std::cerr << "onnx model is already pre-optimized" << std::endl;
        return true;
    }
    preopt_dmutex.unlock_shared();

    auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));
    if (!builder)
    {
        return false;
    }

    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if (!network)
    {
        return false;
    }

    auto config = SampleUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config)
    {
        return false;
    }
        
    config->setMaxWorkspaceSize(4 * size_t(1<<30)); // TODO: configure this
    config->setFlag(nvinfer1::BuilderFlag::kFP16);
    config->setFlag(nvinfer1::BuilderFlag::kREFIT);
    /*config->addOptimizationProfile(profile);*/

	auto parser = SampleUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, sample::gLogger.getTRTLogger()));
	if (!parser)
	{
		return false;
	}

    auto parsed = parser->parse(onnx_model->GetModel(), onnx_model->GetSize());  
    if (!parsed)
    {
        return false;
    }

    auto engine = std::shared_ptr<nvinfer1::ICudaEngine>(
        builder->buildEngineWithConfig(*network, *config), samplesCommon::InferDeleter());
    if (!engine)
    {
        return false;
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    auto serialize = engine->serialize();
    preopt_dmutex.lock();
    preopt_dnns_[onnx_model->name_] = serialize;
    preopt_dmutex.unlock();

    if (save_log_)
    {
        CompileLog *log = new CompileLog;
        log->thread_id = 0; // TODO
        log->task = string("preopt");
        log->model_name = onnx_model->name_;
        log->latency = elapsed.count();
        log_mutex_.lock();
        logs_.push_back(log);
        log_mutex_.unlock();
    }

    return true;
}

void CompileEngine::PreOptHandler()
{
    OptEvent event;
    bool has_event = false;

    while (1)
    {
        preopt_emutex_.lock();
        has_event = false;
        if (!preopt_events_.empty())
        {
            event = preopt_events_.front();
            preopt_events_.pop_front();
            has_event = true;
        }
        preopt_emutex_.unlock();

        if (has_event)
        {
            switch (event.type)
            {
            case OptType::kOpt:
                PreOptInternal(event);
                break;
            case OptType::kJoin:
                return;
            default:
                cerr << "Unsupported event type" << endl;
                break;
            }
        }
    }
}

// bool BuildEngine::Refit(nvinfer1::INetworkDefinition* network, nvinfer1::ICudaEngine* engine)
// bool BuildEngine::Refit(nvinfer1::INetworkDefinition* network, nvinfer1::IHostMemory* trt_model)

// TODO: print a subpixel convolution layer 
static std::pair<std::vector<std::string>, std::vector<nvinfer1::WeightsRole>> getLayerWeightsRolePair(nvinfer1::IRefitter& refitter)
{
    auto const num_All = refitter.getAll(0, nullptr, nullptr);

    // Allocate buffers for the items
    std::vector<const char*> layerNames(num_All);
    std::vector<nvinfer1::WeightsRole> weightsRoles(num_All);
    // Get
    refitter.getAll(num_All, layerNames.data(), weightsRoles.data());
    // container for std::string
    std::vector<std::string> layerNameStrs(num_All);
    
    std::transform(layerNames.begin(), layerNames.end(), layerNameStrs.begin(), [](char const* name)
    {
        if (name == nullptr)
        {
            return std::string{};
        }
        return std::string{name};
    });
    return {layerNameStrs, weightsRoles};
}

std::vector<std::pair<nvinfer1::WeightsRole, nvinfer1::Weights>> getAllRefitWeightsForLayer(const nvinfer1::ILayer& l)
{
    switch (l.getType())
    {
    case nvinfer1::LayerType::kCONSTANT:
    {
        const auto& layer = static_cast<const nvinfer1::IConstantLayer&>(l);
        return {std::make_pair(nvinfer1::WeightsRole::kCONSTANT, layer.getWeights())};
    }
    case nvinfer1::LayerType::kCONVOLUTION:
    {
        const auto& layer = static_cast<const nvinfer1::IConvolutionLayer&>(l);
        return {std::make_pair(nvinfer1::WeightsRole::kKERNEL, layer.getKernelWeights()),
            std::make_pair(nvinfer1::WeightsRole::kBIAS, layer.getBiasWeights())};
    }
    case nvinfer1::LayerType::kDECONVOLUTION:
    {
        const auto& layer = static_cast<const nvinfer1::IDeconvolutionLayer&>(l);
        return {std::make_pair(nvinfer1::WeightsRole::kKERNEL, layer.getKernelWeights()),
            std::make_pair(nvinfer1::WeightsRole::kBIAS, layer.getBiasWeights())};
    }
    case nvinfer1::LayerType::kFULLY_CONNECTED:
    {
        const auto& layer = static_cast<const nvinfer1::IFullyConnectedLayer&>(l);
        return {std::make_pair(nvinfer1::WeightsRole::kKERNEL, layer.getKernelWeights()),
            std::make_pair(nvinfer1::WeightsRole::kBIAS, layer.getBiasWeights())};
    }
    case nvinfer1::LayerType::kSCALE:
    {
        const auto& layer = static_cast<const nvinfer1::IScaleLayer&>(l);
        return {std::make_pair(nvinfer1::WeightsRole::kSCALE, layer.getScale()),
            std::make_pair(nvinfer1::WeightsRole::kSHIFT, layer.getShift())};
    }
    case nvinfer1::LayerType::kRNN_V2:
    case nvinfer1::LayerType::kACTIVATION:
    case nvinfer1::LayerType::kPOOLING:
    case nvinfer1::LayerType::kLRN:
    case nvinfer1::LayerType::kSOFTMAX:
    case nvinfer1::LayerType::kSHUFFLE:
    case nvinfer1::LayerType::kCONCATENATION:
    case nvinfer1::LayerType::kELEMENTWISE:
    case nvinfer1::LayerType::kPLUGIN:
    case nvinfer1::LayerType::kUNARY:
    case nvinfer1::LayerType::kPADDING:
    case nvinfer1::LayerType::kREDUCE:
    case nvinfer1::LayerType::kTOPK:
    case nvinfer1::LayerType::kGATHER:
    case nvinfer1::LayerType::kMATRIX_MULTIPLY:
    case nvinfer1::LayerType::kRAGGED_SOFTMAX:
    case nvinfer1::LayerType::kIDENTITY:
    case nvinfer1::LayerType::kPLUGIN_V2:
    case nvinfer1::LayerType::kSLICE:
    case nvinfer1::LayerType::kFILL:
    case nvinfer1::LayerType::kSHAPE:
    case nvinfer1::LayerType::kPARAMETRIC_RELU:
    case nvinfer1::LayerType::kRESIZE:
    case nvinfer1::LayerType::kTRIP_LIMIT:
    case nvinfer1::LayerType::kRECURRENCE:
    case nvinfer1::LayerType::kITERATOR:
    case nvinfer1::LayerType::kLOOP_OUTPUT:
    case nvinfer1::LayerType::kSELECT:
    default:
     return {};
    // case nvinfer1::LayerType::kQUANTIZE:
    // case nvinfer1::LayerType::kDEQUANTIZE: return {};
    }
    return {};
}

 std::pair<std::vector<std::string>, std::vector<nvinfer1::WeightsRole>> getMissingLayerWeightsRolePair(nvinfer1::IRefitter& refitter)
{
    auto const num_Missing = refitter.getMissing(0, nullptr, nullptr);

    // Allocate buffer for the items
    std::vector<const char*> layerNames(num_Missing);
    std::vector<nvinfer1::WeightsRole> weightsRoles(num_Missing);  
    // Get
    refitter.getMissing(num_Missing, layerNames.data(), weightsRoles.data());
    // container for std::string
    std::vector<std::string> layerNameStrs(num_Missing);

    std::transform(layerNames.begin(), layerNames.end(), layerNameStrs.begin(), [](char const* name)
    {
        if (name == nullptr)
        {
            return std::string{};
        }
        return std::string{name};
    });
    return {layerNameStrs, weightsRoles};
}

// TODO: measure deserialization latency
nvinfer1::IHostMemory *CompileEngine::Refit(nvinfer1::INetworkDefinition* network, nvinfer1::IHostMemory* trt_model)
{
    SampleUniquePtr<IRuntime> runtime{createInferRuntime(sample::gLogger.getTRTLogger())};
    if (!runtime)
    {
        return nullptr;
    }
    auto engine = std::shared_ptr<nvinfer1::ICudaEngine>(
        runtime->deserializeCudaEngine(trt_model->data(), trt_model->size()), samplesCommon::InferDeleter());
    if (!engine)
    {
        return nullptr;
    }

    auto start = std::chrono::high_resolution_clock::now();

    auto const num_Layers = network->getNbLayers();
    // TODO: unique_ptr, glogger
    SampleUniquePtr<nvinfer1::IRefitter> refitter{nvinfer1::createInferRefitter(*engine, sample::gLogger.getTRTLogger())};
    auto const& layerWeightsRolePair = getLayerWeightsRolePair(*refitter);
    
    // used for checking existence
    std::set<std::pair<std::string, nvinfer1::WeightsRole>> layerRoleSet;
    auto const& layerNames = layerWeightsRolePair.first;
    auto const& weightsRoles = layerWeightsRolePair.second;

    // fill the set
    std::transform(layerNames.begin(), layerNames.end(), weightsRoles.begin(), 
        std::inserter(layerRoleSet, layerRoleSet.begin()),
        [](std::string const& layerName, nvinfer1::WeightsRole const& role) { return std::make_pair(layerName, role); });

    // function to check is refittable
    auto const isRefittable = [&layerRoleSet](char const* layerName, nvinfer1::WeightsRole const role) {
        return layerRoleSet.find(std::make_pair(layerName, role)) != layerRoleSet.end();
    };

    // function to refit the engine using network weights
    auto const setWeights = [&] {
        for (int32_t i=0; i < num_Layers; i++)
        {
            auto const layer = network->getLayer(i);
            auto const rolWeightsVec = getAllRefitWeightsForLayer(*layer);
            for (auto const& roleWeights : rolWeightsVec)
            {
                if (isRefittable(layer->getName(), roleWeights.first))
                {
                    bool const success = refitter->setWeights(layer->getName(), roleWeights.first, roleWeights.second);
                    if (!success)
                    {
                        return false;
                    }
                }
            }
        }
        return true;
    };

    auto const reportMissingWeights = [&] {
        auto const& missingPair = getMissingLayerWeightsRolePair(*refitter);
        auto const& layerNames = missingPair.first;
        // auto const& weightsRoles = missingPair.second;
        for (size_t i = 0; i < layerNames.size(); i++)
        {
            sample::gLogError << "Missing (" << layerNames[i] << ") for refitting." << std::endl;
        }
        return layerNames.empty();
    };

    // Do refitting and check missing weights
    IHostMemory *preopt_dnn;
    // bool const success = setWeights() && reportMissingWeights() && refitter->refitCudaEngine();
    bool const success = setWeights() && reportMissingWeights() && refitter->refitCudaEngine();
    if (!success)
    {
        return nullptr;
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Latency (Refit): " << elapsed.count() * 1000 << "ms" << std::endl;

    preopt_dnn = engine->serialize();

    return preopt_dnn;
}


bool CompileEngine::OptInternal(OptEvent &event, int thread_id)
{
    OnnxModel *onnx_model = event.onnx_model;
    int stream_id = event.stream_id;

    if (!onnx_model->is_loaded_)
    {
        std::cerr << "onnx model is not loaded" << std::endl;
        return false;
    }
 
    preopt_dmutex.lock_shared();
    if (preopt_dnns_.find(onnx_model->name_) == preopt_dnns_.end())
    {
        std::cerr << "pre-optimized onnx model does not exist" << std::endl;
        preopt_dmutex.unlock_shared();
        return false;
    }
    IHostMemory *preopt_dnn = preopt_dnns_[onnx_model->name_];
    preopt_dmutex.unlock_shared();


    auto start = std::chrono::high_resolution_clock::now();

    auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));
    if (!builder)
    {
        return false;
    }

    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if (!network)
    {
        return false;
    }

    auto parser = SampleUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, sample::gLogger.getTRTLogger()));
	if (!parser)
	{
		return false;
	}

    auto parsed = parser->parse(onnx_model->GetModel(), onnx_model->GetSize());  
    if (!parsed)
    {
        return false;
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Latency (parse): " << elapsed.count() << std::endl;

    IHostMemory *opt_dnn = Refit(network.get(), preopt_dnn);
    if (opt_dnn == nullptr)
    {
        sample::gLogError << "Refit failed" << std::endl;
        return false;
    }



    OptDNN *result = new OptDNN;
    result->stream_id = stream_id;
    result->opt_dnn = opt_dnn;

    opt_dmutex_.lock();
    // opt_dnns_[stream_id] = opt_dnn;
    opt_dnns_.push_back(result);
    opt_dmutex_.unlock();

    if (save_log_)
    {
        CompileLog *log = new CompileLog;
        log->thread_id = thread_id; // TODO
        log->task = string("opt");
        log->model_name = onnx_model->name_;
        log->latency = elapsed.count();
        log_mutex_.lock();
        logs_.push_back(log);
        log_mutex_.unlock();
    }

    return true;
}


void CompileEngine::OptHandler(int thread_id)
{
    OptEvent event;
    bool has_event = false;

    while (1)
    {
        opt_emutex_.lock();
        has_event = false;
        if (!opt_events_.empty())
        {
            event = opt_events_.front();
            opt_events_.pop_front();
            has_event = true;
        }
        opt_emutex_.unlock();

        if (has_event)
        {
            switch (event.type)
            {
            case OptType::kOpt:
                OptInternal(event, thread_id);
                // TODO: call optinternal()
                break;
            case OptType::kJoin:
                return;
            default:
                cerr << "Unsupported event type" << endl;
                break;
            }
        }
    }
}

void CompileEngine::PreOptimize(OnnxModel *onnx_model)
{
    OptEvent event;
    event.type = OptType::kOpt;
    event.onnx_model = onnx_model;

    preopt_emutex_.lock();
    preopt_events_.push_back(event);
    preopt_emutex_.unlock();
}

// TODO: measure latency 
// nvinfer1::IHostMemory* CompileEngine::Optimize(int stream_id, OnnxModel *onnx_model)
void CompileEngine::Optimize(int stream_id, OnnxModel *onnx_model)
{
    OptEvent event;
    event.stream_id = stream_id;
    event.type = OptType::kOpt;
    event.onnx_model = onnx_model;

    opt_emutex_.lock();
    opt_events_.push_back(event);
    opt_emutex_.unlock();

    // IHostMemory *opt_dnn = nullptr;
    // bool found = false;
    // opt_dmutex_.lock();
    // while (!found)
    // {
    //     if (opt_dnns_.find(stream_id) != opt_dnns_.end())
    //     {
    //         opt_dnn = opt_dnns_[stream_id];
    //         opt_dnns_.erase(stream_id);
    //         found = true;
    //     }
    //     opt_dmutex_.unlock();
    // }
    
    // return opt_dnn;
}

void CompileEngine::GetOptDNNs(std::deque<OptDNN*> &opt_dnns)
{
    OptDNN *opt_dnn;

    opt_dmutex_.lock();
    while (opt_dnns_.size() > 0)
    {
        opt_dnn = opt_dnns_.front();
        opt_dnns_.pop_front();
        opt_dnns.push_back(opt_dnn);
    }

    opt_dmutex_.unlock();
}

int CompileEngine::GetOptSize()
{
    return opt_dnns_.size();
}

bool CompileEngine::PreOptExists(std::string dnn_name)
{
    preopt_dmutex.lock_shared();
    if (preopt_dnns_.find(dnn_name) == preopt_dnns_.end())
    {
        preopt_dmutex.unlock_shared();
        return false;
    }
    preopt_dmutex.unlock_shared();
    return true;
}

// void CompileEngine::Register(std::string &name, nvinfer1::IHostMemory* preopt_dnn)
void CompileEngine::Register(std::string &name, void *buf, size_t size)
{
    auto start = std::chrono::high_resolution_clock::now();

    preopt_dmutex.lock_shared();
    if (preopt_dnns_.find(name) != preopt_dnns_.end())
    {
        return;
    }
    preopt_dmutex.unlock_shared();

    auto runtime = SampleUniquePtr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(sample::gLogger.getTRTLogger()));
    if (!runtime)
    {
        return;
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    auto engine = std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(buf, size), 
                            samplesCommon::InferDeleter());
    if (!engine)
    {
        return;
    }   

    auto preopt_dnn = engine->serialize();
    // std::cout << elapsed.count() << std::endl;


    preopt_dmutex.lock();
    if (preopt_dnns_.find(name) == preopt_dnns_.end())
    {
        preopt_dnns_[name] = preopt_dnn;
    }
    preopt_dmutex.unlock();

    if (save_log_)
    {
        CompileLog *log = new CompileLog;
        log->thread_id = 0; // TODO
        log->task = string("regist");
        log->model_name = name;
        log->latency = elapsed.count();
        log_mutex_.lock();
        logs_.push_back(log);
        log_mutex_.unlock();
    }

}

void CompileEngine::Save()
{
    if (!save_log_)
        return;

    // set base dir and create it 
    if (log_dir_.empty())
    {
        // default dir:  ([binary dir]/results/anchor_engine/[year-month-date]/...)
        std::string today = date::format("%F", std::chrono::system_clock::now());   
        // log_dir_ = fs::current_path() / "results" / "compile_engine" / today;
        log_dir_ = fs::current_path() / "results" / "compile_engine";

    }
    if (!fs::exists(log_dir_))
        fs::create_directories(log_dir_);
       
    // save logs
    fs::path log_path = log_dir_ / "latency.txt";
    std::ofstream log_file;
    std::string log_str;

    log_file.open(log_path);
    if (log_file.is_open())
    {
        log_mutex_.lock();
        for (auto log : logs_)
        {
            log_str = log->task + '\t' + to_string(log->thread_id) + '\t' + log->model_name + '\t' + to_string(log->latency) + '\n';
            log_file << log_str;
        }
        log_mutex_.unlock();
    }
}