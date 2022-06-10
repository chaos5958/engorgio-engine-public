/**
 * Tester for cuHostMalloc.
 * This test randomly allocate and free pointer,
 * and check validity of each pointer.
*/

#include <iostream>
#include <thread>
#include <vector>
#include <chrono>
#include <algorithm>
#include <string.h>
#include <random>
#include "cuHostMemory.h"

std::vector<std::thread> threads;
int* idx; uint8_t* randomArray;

/** Genertate \p size random uint8_t array */
void randGen(uint8_t* p, size_t size, int token)
{
    std::mt19937 gen(token);
    std::uniform_int_distribution<int> dis(0, 255);

    for (size_t i = 0; i < size; i++) {
        p[i] = dis(gen);
    }
}

/** Return true if \p size uint8_t array \p p has value \p val. False otherwise. */
inline bool checkArray(uint8_t* p, uint8_t val, size_t size)
{
    for (size_t i = 0; i < size; i++) {
        if (p[i] != val)
            return false;
    }

    return true;
}

bool test(int numTest, int minSize, int fragment, int threadId)
{
    cuHostMemory* hmemory = cuHostMemory::GetInstance(minSize, fragment);
    void** ptr;
    int size;

    ptr = (void**)calloc(numTest, sizeof(void*));
    if (ptr == nullptr)
        return false;

    for (int i = 0; i < numTest * 2; i++) {
        // Randomly allocate and free pointer.
        if (ptr[idx[i]] == nullptr) {
            size = minSize << (randomArray[idx[i]] % fragment);
            ptr[idx[i]] = hmemory->Malloc(size);
            memset(ptr[idx[i]], randomArray[idx[i]], size);
        }
        else {
            hmemory->Free(ptr[idx[i]]);
            ptr[idx[i]] = nullptr;
        }

        // Check validity of each pointer.
        if (i % 128 == 0) {
            for (int j = 0; j < numTest; j++) {
                if (ptr[j] != nullptr) {
                    size = minSize << (randomArray[j] % fragment);
                    if (!checkArray((uint8_t*)ptr[j], randomArray[j], size))
                        return false;
                    memset(ptr[j], randomArray[j], size);
                }
            }
        }
    }

    std::cout << "success!" << threadId << std::endl;
    return true;
}

int main(int argc, char* argv[])
{
    int numTest = 1024 * 4, minSize = 1024 * 31, fragment = 7, nThread = 16;

    idx = (int*)malloc(numTest * 2 * sizeof(int));
    randomArray = (uint8_t*)malloc(numTest);

    if (idx == nullptr || randomArray == nullptr)
        return 1;

    // Initialize
    for (int i = 0; i < numTest; i++) {
        idx[i] = i;
    }
    for (int i = 0; i < numTest; i++) {
        idx[i + numTest] = i;
    }

    std::random_shuffle(idx, idx + numTest * 2);
    randGen((uint8_t*)randomArray, numTest, 64);

    std::cout << "prepared!" << std::endl;
    for (int i = 0; i < nThread; i++)
        threads.push_back(std::thread(test, numTest, minSize - 1, fragment, i));

    for (auto& t : threads) {
        t.join();
    }

    return 0;
}