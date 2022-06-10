#include <stdio.h>
#include<string.h>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <chrono>
#include <thread>
#include "ipp.h"
#include "ippcc.h"
// #include "ippcore.h"
// #include "ippvm.h"
// #include "ipps.h"
#include "ippi.h"

#define PRINT_INFO(feature, text) printf("  %-30s= ", #feature); \
      printf("%c\t%c\t", (cpuFeatures & feature) ? 'Y' : 'N', (enabledFeatures & feature) ? 'Y' : 'N'); \
      printf( #text "\n")

int hellow_word()
{
      const       IppLibraryVersion *libVersion;
      IppStatus   status;
      Ipp64u      cpuFeatures, enabledFeatures;

      ippInit();                      /* Initialize Intel® IPP library */
      libVersion = ippGetLibVersion(); /* Get Intel® IPP library version info */
      printf("%s %s\n", libVersion->Name, libVersion->Version);

      status = ippGetCpuFeatures(&cpuFeatures, 0); /* Get CPU features and features enabled with selected library level */
      if (ippStsNoErr != status) return status;
      enabledFeatures = ippGetEnabledCpuFeatures();
      printf("Features supported: by CPU\tby Intel® IPP\n");
      printf("------------------------------------------------\n");
      PRINT_INFO(ippCPUID_MMX,        Intel® Architecture MMX technology supported);
      PRINT_INFO(ippCPUID_SSE,        Intel® Streaming SIMD Extensions);
      PRINT_INFO(ippCPUID_SSE2,       Intel® Streaming SIMD Extensions 2);
      PRINT_INFO(ippCPUID_SSE3,       Intel® Streaming SIMD Extensions 3);
      PRINT_INFO(ippCPUID_SSSE3,      Supplemental Streaming SIMD Extensions 3);
      PRINT_INFO(ippCPUID_MOVBE,      Intel® MOVBE instruction);
      PRINT_INFO(ippCPUID_SSE41,      Intel® Streaming SIMD Extensions 4.1);
      PRINT_INFO(ippCPUID_SSE42,      Intel® Streaming SIMD Extensions 4.2);
      PRINT_INFO(ippCPUID_AVX,        Intel® Advanced Vector Extensions instruction set);
      PRINT_INFO(ippAVX_ENABLEDBYOS,  Intel® Advanced Vector Extensions instruction set is supported by OS);
      PRINT_INFO(ippCPUID_AES,        Intel® AES New Instructions);
      PRINT_INFO(ippCPUID_CLMUL,      Intel® CLMUL instruction);
      PRINT_INFO(ippCPUID_RDRAND,     Intel® RDRAND instruction);
      PRINT_INFO(ippCPUID_F16C,       Intel® F16C new instructions);
      PRINT_INFO(ippCPUID_AVX2,       Intel® Advanced Vector Extensions 2 instruction set);
      PRINT_INFO(ippCPUID_ADCOX,      Intel® ADOX/ADCX new instructions);
      PRINT_INFO(ippCPUID_RDSEED,     Intel® RDSEED instruction);
      PRINT_INFO(ippCPUID_PREFETCHW,  Intel® PREFETCHW instruction);
      PRINT_INFO(ippCPUID_SHA,        Intel® SHA new instructions);
      PRINT_INFO(ippCPUID_AVX512F,    Intel® Advanced Vector Extensions 512 Foundation instruction set);
      PRINT_INFO(ippCPUID_AVX512CD,   Intel® Advanced Vector Extensions 512 CD instruction set);
      PRINT_INFO(ippCPUID_AVX512ER,   Intel® Advanced Vector Extensions 512 ER instruction set);
      PRINT_INFO(ippCPUID_AVX512PF,   Intel® Advanced Vector Extensions 512 PF instruction set);
      PRINT_INFO(ippCPUID_AVX512BW,   Intel® Advanced Vector Extensions 512 BW instruction set);
      PRINT_INFO(ippCPUID_AVX512VL,   Intel® Advanced Vector Extensions 512 VL instruction set);
      PRINT_INFO(ippCPUID_AVX512VBMI, Intel® Advanced Vector Extensions 512 Bit Manipulation instructions);
      PRINT_INFO(ippCPUID_MPX,        Intel® Memory Protection Extensions);
      PRINT_INFO(ippCPUID_AVX512_4FMADDPS,    Intel® Advanced Vector Extensions 512 DL floating-point single precision);
      PRINT_INFO(ippCPUID_AVX512_4VNNIW,      Intel® Advanced Vector Extensions 512 DL enhanced word variable precision);
      PRINT_INFO(ippCPUID_KNC,        Intel® Xeon Phi™ Coprocessor);
      PRINT_INFO(ippCPUID_AVX512IFMA, Intel® Advanced Vector Extensions 512 IFMA (PMADD52) instruction set);
      PRINT_INFO(ippAVX512_ENABLEDBYOS,       Intel® Advanced Vector Extensions 512 is supported by OS);
      return 0;
}

int benchmark_yuv2rgb(int width, int height, int num_repeats)
{
      Ipp8u* y_buf = (Ipp8u*) malloc(width * height * sizeof(Ipp8u));
      Ipp8u* u_buf = (Ipp8u*) malloc(width * height * sizeof(Ipp8u));
      Ipp8u* v_buf = (Ipp8u*) malloc(width * height * sizeof(Ipp8u));
      const Ipp8u* yuv_buf[3] = {y_buf, u_buf, v_buf};
      int y_stride = width;
      int uv_stride = width / 2;
      int yuv_stride[3] = {y_stride, uv_stride, uv_stride};
      IppiSize roi_size = {width, height};
      Ipp8u* rgb_buf = (Ipp8u*) malloc(width * height * 3 * sizeof(Ipp8u));
      int rgb_stride = width * 3;
      Ipp8u alpha = 0xFF;

      IppStatus st = ippStsNoErr;
      
      auto start = std::chrono::high_resolution_clock::now();
      for (int i = 0; i < num_repeats; i++)
      {
            st = ippiYCbCr420ToBGR_709HDTV_8u_P3C4R(yuv_buf, yuv_stride, rgb_buf, rgb_stride, roi_size, alpha);
            if ( st != ippStsNoErr)
            {
                  std::cout << "failed: " << st << std::endl;
                  return -1;
            }
      }
      auto end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> elapsed = end - start;
      std::cout << "Latency (YUV2RGB): " << elapsed.count() * 1000 / num_repeats << "ms" << std::endl;
      return 0;
}

int benchmark_rgb2yuv(int width, int height, int num_repeats)
{
      Ipp8u* y_buf = (Ipp8u*) malloc(width * height * sizeof(Ipp8u));
      Ipp8u* u_buf = (Ipp8u*) malloc(width * height * sizeof(Ipp8u));
      Ipp8u* v_buf = (Ipp8u*) malloc(width * height * sizeof(Ipp8u));
      Ipp8u* yuv_buf[3] = {y_buf, u_buf, v_buf};
      int y_stride = width;
      int uv_stride = width / 2;
      int yuv_stride[3] = {y_stride, uv_stride, uv_stride};
      IppiSize roi_size = {width, height};
      const Ipp8u* rgb_buf = (Ipp8u*) malloc(width * height * 3 * sizeof(Ipp8u));
      int rgb_stride = width * 3;
      Ipp8u alpha = 0xFF;

      IppStatus st = ippStsNoErr;
      
      auto start = std::chrono::high_resolution_clock::now();
      for (int i = 0; i < num_repeats; i++)
      {
            // auto start = std::chrono::high_resolution_clock::now();
            st = ippiBGRToYCbCr420_709HDTV_8u_AC4P3R(rgb_buf, rgb_stride, yuv_buf, yuv_stride, roi_size);
            if ( st != ippStsNoErr)
            {
                  std::cout << "failed: " << st << std::endl;
                  return -1;
            }
            // auto end = std::chrono::high_resolution_clock::now();
            // std::chrono::duration<double> elapsed = end - start;
            // std::cout << "Latency (YUV2RGB): " << elapsed.count() * 1000  << "ms" << std::endl;
            // std::this_thread::sleep_for(std::chrono::milliseconds(1000));
      }
      auto end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> elapsed = end - start;
      std::cout << "Latency (RGB2YUV): " << elapsed.count() * 1000 / num_repeats << "ms" << std::endl;
      return 0;
}

void read_file(std::string path, Ipp8u **data)
{
	std::ifstream is(path, std::ifstream::binary);
	if (is) {
		// seekg를 이용한 파일 크기 추출
		is.seekg(0, is.end);
		int length = (int)is.tellg();
		is.seekg(0, is.beg);

		// malloc으로 메모리 할당
		Ipp8u* buffer = (Ipp8u*)malloc(length);
            memset(buffer, 0, length);

		// read data as a block:
		is.read((char*)buffer, length);
		is.close();
		*data = buffer; 
	}
}

void write_file(std::string path, Ipp8u *data, int len)
{
      std::ofstream fout;
	fout.open(path, std::ios::out | std::ios::binary);
    
	if (fout.is_open()){
		fout.write((const char*)data, len);
		fout.close();
	}
}

void save_yuv2rgb()
{
      // settings
      int width = 1280;
      int height = 720;
      int num_frames = 15;
      std::string img_dir = "/workspace/research/engorgio-dataset/game1/image/720p_3960kbps_s960_d60.webm/key";

      // setup
      std::vector<Ipp8u*> y_buffers, u_buffers, v_buffers;
      int y_stride = width, uv_stride = width / 2 ;
      int yuv_stride[3] = {y_stride, uv_stride, uv_stride};
      std::vector<Ipp8u*> rgb_buffers;
      int rgb_stride = width * 3;
      IppiSize roi_size = {width, height};

      // alloc rgb frames
      Ipp8u* rgb_buffer;
      for (int i = 0; i < num_frames; i++)
      {
            rgb_buffer = (Ipp8u*) malloc(width * height * 3 * sizeof(Ipp8u));
            memset(rgb_buffer, 0, width * height * 3 * sizeof(Ipp8u));
            rgb_buffers.push_back(rgb_buffer);
      }
      // load yuv frames
      std::string img_path;
      Ipp8u* y_buffer, *u_buffer, *v_buffer;
      char buffer[256];
      for (int i = 1; i <= num_frames; i++)
      {
            sprintf(buffer, "%04d", i);
            img_path = img_dir + "/" + std::string(buffer) + ".y";
            std::cout << img_path << std::endl;
            read_file(img_path, &y_buffer);
            y_buffers.push_back(y_buffer);
            img_path = img_dir + "/" + std::string(buffer) + ".u";
            std::cout << img_path << std::endl;
            read_file(img_path, &u_buffer);
            u_buffers.push_back(u_buffer);
            img_path = img_dir + "/" + std::string(buffer) + ".v";
            std::cout << img_path << std::endl;
            read_file(img_path, &v_buffer);
            v_buffers.push_back(v_buffer);
      }

      std::cout << y_buffers.size() << std::endl;
      std::cout << u_buffers.size() << std::endl;
      std::cout << v_buffers.size() << std::endl;
      // apply yuv to rgb conversion
      // TODO: start from here
      IppStatus st = ippStsNoErr;      
      const Ipp8u* yuv_buf[3];
      auto start = std::chrono::high_resolution_clock::now();
      const int rgb_order[3] = {2, 1, 0};
      for (int i = 0; i < num_frames; i++)
      {
            yuv_buf[0] = y_buffers[i];
            yuv_buf[1] = u_buffers[i];
            yuv_buf[2] = v_buffers[i];
            st = ippiYCbCr420ToBGR_709CSC_8u_P3C3R(yuv_buf, yuv_stride, rgb_buffers[i], rgb_stride, roi_size);
            if ( st != ippStsNoErr)
            {
                  std::cout << "failed: " << st << std::endl;
                  return;
            }

            st = ippiSwapChannels_8u_C3R(rgb_buffers[i], rgb_stride, rgb_buffers[i], rgb_stride, roi_size, rgb_order);
            if ( st != ippStsNoErr)
            {
                  std::cout << "failed: " << st << std::endl;
                  return;
            }
      }
      auto end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> elapsed = end - start;
      std::cout << "Latency (YUV2RGB): " << elapsed.count() * 1000 / 15 << "ms" << std::endl;

      // save rgb frames (*.rgb)
      for (int i = 1; i <= num_frames; i++)
      {
            sprintf(buffer, "%04d", i);
            img_path = img_dir + "/" + std::string(buffer) + ".rgb";  
            write_file(img_path, rgb_buffers[i-1], width * height * 3);
      }

      // free allocated memory   
      for (int i = 0; i < num_frames; i++)
      {
            free(y_buffers[i]);
            free(u_buffers[i]);
            free(v_buffers[i]);
            free(rgb_buffers[i]);
      }
}

int main(int argc, char* argv[])
{
      int width, height;
      int num_repeats = 100;

      ippInit();    
      
      width = 1280;
      height = 720;
      // benchmark_yuv2rgb(width, height, num_repeats);

      width = 3840;
      height = 2160;
      // benchmark_rgb2yuv(width, height, num_repeats);

      save_yuv2rgb();

      return 0;
}
