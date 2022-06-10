#!/bin/bash

cmake --no-warn-unused-cli -DCMAKE_EXPORT_COMPILE_COMMANDS:BOOL=TRUE -DCMAKE_BUILD_TYPE:STRING=Debug -DCMAKE_C_COMPILER:FILEPATH=/bin/x86_64-linux-gnu-gcc-9 -DCMAKE_CXX_COMPILER:FILEPATH=/bin/x86_64-linux-gnu-g++-9 -H/workspace/research/engine -B/workspace/research/engine/build -G Ninja

