cmake_minimum_required(VERSION 3.22.1)

project("ainamaura_engine")

# NDK NNAPI is available from API 27+
find_library(log-lib log)

# Include Neural Networks API (NNAPI)
# Note: For INT4, NNAPI has support via ANeuralNetworksOperandType but often delegates
# to CPU/NPU. We will structure the code to use standard vectors for the unified loop
# and delegate to NNAPI where possible.

add_library(
        ainamaura_engine
        SHARED
        ainamaura_engine.cpp
)

target_link_libraries(
        ainamaura_engine
        ${log-lib}
        # android
        # neuralnetworks
)
