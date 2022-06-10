#include "compile_engine.h"
#include "tool_common.h"

void test_preopt(OnnxModel *model)
{
    CompileEngineProfile cprofile;
    cprofile.num_threads = 0;
    cprofile.save_log = true;

    CompileEngine *cengine = new CompileEngine(cprofile);
    cengine->PreOptimize(model);
    delete cengine;
}

static size_t read_file(const std::string& file, void** buffer)
{
    size_t size;

    std::ifstream stream(file, std::ifstream::ate | std::ifstream::binary);
    if (!stream.is_open())
    {
        return 0;
    }

    size = stream.tellg();
    stream.seekg(0, std::ios::beg);
    *buffer = (unsigned char*)malloc(size);
    if (*buffer == nullptr)
        return 0;

    stream.read((char*)*buffer, size);
    return size;
}

void test_opt(OnnxModel *model, std::string& plan_path, int num_workers, int num_repeats)
{
    CompileEngineProfile cprofile;
    cprofile.num_threads = num_workers;
    cprofile.save_log = true;
    void *buf {nullptr};
    size_t size;

    CompileEngine *cengine = new CompileEngine(cprofile);

    if (plan_path.length() == 0)
    {
        cengine->PreOptimize(model);
        while (!cengine->PreOptExists(model->name_))
        {}
    }
    else
    {
        size = read_file(plan_path, &buf);
        if (size == 0)
        {
            delete cengine;
            return;
        }
        // std::cout << "size: " << size << std::endl;
        cengine->Register(model->name_, buf, size);
    }

    for (int i = 0; i < num_repeats; i++)
    {
        cengine->Optimize(i, model);
    }
    while (cengine->GetOptSize() != num_repeats)
    {}

    std::deque<OptDNN*> opt_dnns;
    cengine->GetOptDNNs(opt_dnns);
    while(opt_dnns.size() > 0)
    {
        opt_dnns.front()->opt_dnn->destroy();
        opt_dnns.pop_front();
    }

    cengine->Save();
    delete cengine;
}

// TODO: refactor, now we use refit() instead of deserializing a cuda engine
int main()
{
    // std::string content = "animation1";
    // int resolution = 360;
    // int duration = xxx;
    // std::string model = "EDSR_B8_F32_S3";
    // std::string video = get_video_name(resolution, duration);
    // std::string onnx_path = data_dir + "/" + content + "/" + "checkpoint" + "/" + video + "/" + model + "/" + model + ".onnx";
    // std::string plan_path;
    // OnnxModel *onnx_model = new OnnxModel(onnx_path, std::string("tmp"));
    // if (!onnx_model->Load())
    //     exit(0);

    // // test_preopt(onnx_model);
    // // test_opt(onnx_model, plan_path, 5, 50);

    // plan_path = data_dir + "/" + content + "/" + "checkpoint" + "/" + video + "/" + model + "/" + model + ".plan";
    // // test_opt(onnx_model, plan_path, 5, 50);
    // test_opt(onnx_model, plan_path, 1, 10);


    // delete onnx_model;
    // std::cout << "Test finished" << std::endl;

    // return 0;
}