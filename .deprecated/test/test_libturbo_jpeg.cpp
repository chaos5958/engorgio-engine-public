#include <iostream>
#include <thread>
#include <sys/time.h>
#include <cstring>
#include <cassert>
#include <turbojpeg.h>
#include <chrono>
#include <stdlib.h>
#include <unistd.h>
#include <cstring>
#include <sys/wait.h>

#define CHECK(x)                                                                  \
    if (!(x))                                                                     \
    {                                                                             \
        fprintf(stderr, "Assertion Failed: [%s:%d] %s", __FILE__, __LINE__, #x); \
        abort();                                                                  \
    }

struct ImageMetadata {
    uint32_t width;
    uint32_t height;
};

double get_psnr(const char *fn1, const char *fn2) {
    int link[2];
    pid_t pid;
    char foo[128] = {0, };

    CHECK(pipe(link) >= 0);
    CHECK((pid = fork()) >= 0)

    if (pid == 0) {
        dup2 (link[1], STDOUT_FILENO);
        close(link[0]);
        close(link[1]);
        execl("/usr/bin/env", "python", "/workspace/research/engorgio-engine/test/calculate_psnr.py", fn1, fn2, (char *)0);
    } else {
        close(link[1]);
        int r = read(link[0], foo, sizeof(foo));
        ((void) r);

        wait(NULL);
    }
    return atof(foo);
}

int main(int argc, const char *argv[]) {

    const TJSAMP OPTION_SUBSMPL[] = {
        TJSAMP_444, TJSAMP_420
    };

    const int OPTION_QP[] = {
        65, 70, 75, 80, 85, 90
        // , 80, 85, 90, 95, 100
    };

    const int OPTION_FLAGS[] = {
        TJFLAG_FASTDCT,
        // TJFLAG_ACCURATEDCT,
        // TJFLAG_PROGRESSIVE | TJFLAG_ACCURATEDCT
    };

    if (argc < 3) {
        fprintf(stderr, "Usage: %s [dir] [num_images]\n", argv[0]);
    }
    const char *image_dir = argv[1];
    unsigned num_images = atoi(argv[2]);

    FILE *file_stat = stdout;
    fprintf(file_stat, "Filename,width,height,raw_size,subsampling,qp,flags,elapsed_us,fps,output_size,psnr\n");

    for (unsigned image_id = 1; image_id <= num_images; image_id++) {
        char *filename = new char [strlen(image_dir) + 10]();
        char *png_filename = new char [strlen(image_dir) + 10]();
        char *meta_filename = new char [strlen(image_dir) + 10]();
        sprintf(filename, "%s/%.4d.raw", image_dir, image_id);
        sprintf(png_filename, "%s/%.4d.png", image_dir, image_id);
        sprintf(meta_filename, "%s/%.4d.meta", image_dir, image_id);

        std::cerr << filename << std::endl;
        FILE *raw_file, *meta_file;
        unsigned long raw_file_len = 0, meta_file_len = 0;
        unsigned char *src_buf, *dst_buf;
        struct ImageMetadata metadata;

        // open file
        CHECK(raw_file = fopen(filename, "rb"));
        if (fseek(raw_file, 0, SEEK_END) < 0 || ((raw_file_len = ftell(raw_file)) < 0) || fseek(raw_file, 0, SEEK_SET) < 0) {
            std::cerr << "DETERMINE INPUT FILE SIZE" << std::endl;
            continue;
        }
        CHECK(meta_file = fopen(meta_filename, "rb"));
        if (fseek(meta_file, 0, SEEK_END) < 0 || ((meta_file_len = ftell(meta_file)) < 0) || fseek(meta_file, 0, SEEK_SET) < 0) {
            std::cerr << "DETERMINE INPUT FILE SIZE" << std::endl;
            continue;
        }

        CHECK(raw_file_len > 0);
        CHECK(meta_file_len >= sizeof(struct ImageMetadata));

        src_buf = new unsigned char[raw_file_len];
        dst_buf = tjAlloc(raw_file_len);
        CHECK(fread(src_buf, raw_file_len, 1, raw_file) > 0);
        CHECK(fread(&metadata, sizeof(struct ImageMetadata), 1, meta_file) > 0);

        // compression
        tjhandle encodeInstance;
        CHECK(encodeInstance = tjInitCompress());

        unsigned long jpg_file_len = raw_file_len;

        for (unsigned i = 0; i < sizeof(OPTION_SUBSMPL) / sizeof(TJSAMP); i++) {
            for (unsigned j = 0; j < sizeof(OPTION_QP) / sizeof(int); j++) {
                for (unsigned k = 0; k < sizeof(OPTION_FLAGS) / sizeof(int); k++) {
                    const TJSAMP option_subsampl = OPTION_SUBSMPL[i];
                    const int option_qp = OPTION_QP[j];
                    const int option_flags = OPTION_FLAGS[k];

                    auto time_begin = std::chrono::steady_clock::now();
                    CHECK(encodeInstance = tjInitCompress());
                    CHECK(tjCompress2(encodeInstance, src_buf, metadata.width, 0, metadata.height, TJPF_RGB, &dst_buf, &jpg_file_len, option_subsampl, option_qp, option_flags) == 0);
                    auto time_end = std::chrono::steady_clock::now();
                    unsigned long elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_begin).count();
                    
                    // save image
                    char *dst_filename = new char [strlen(filename) + 128]();
                    sprintf(dst_filename, "%s.s%dq%df%d.jpg", png_filename, option_subsampl, option_qp, option_flags);
                    FILE *dst_file = fopen(dst_filename, "w");

                    std::string hr_filename = std::string(png_filename);
                    hr_filename = hr_filename.replace(hr_filename.find("/sr/"), std::string("/sr/").length(), "/hr/");
                    CHECK(fwrite(dst_buf, jpg_file_len, 1, dst_file) == 1);
                    fflush(dst_file);
                    fclose(dst_file);

                    // get PSNR
                    double psnr = get_psnr(hr_filename.c_str(), dst_filename);
                    delete[] dst_filename;

                    fprintf(stderr, "%s (%u x %u / %.3lf MB): Subsampling=%d, QP=%d, Flags=%d -> Took %luus (%.3lffps), size %.3lfKB, PSNR %.3lfdB\n", 
                            filename, metadata.width, metadata.height, (double(raw_file_len)/1000000.), option_subsampl, option_qp, option_flags, 
                            elapsed_us, 1000000./elapsed_us, (double(jpg_file_len)/1000.), psnr);
                    fprintf(file_stat, "\"%s\",%u,%u,%lu,%d,%d,%d,%lu,%lf,%lu,%lf\n",
                            filename, metadata.width, metadata.height, raw_file_len, option_subsampl, option_qp, option_flags, 
                            elapsed_us, 1000000./elapsed_us, jpg_file_len, psnr);
                }
            }
        }
        
        tjDestroy(encodeInstance);
        tjFree(dst_buf);
        delete[] src_buf;
        delete[] filename;
        delete[] png_filename;
        delete[] meta_filename;
    }
}