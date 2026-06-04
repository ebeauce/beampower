cmake_minimum_required(VERSION 3.24)
project(beampower LANGUAGES C CXX)

# 1. Initialize Python Extension Build Support
find_package(Python REQUIRED COMPONENTS Development.Module)

# 2. Check for CUDA support dynamically
include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    message(STATUS "CUDA found! GPU support will be compiled.")
    set(HAS_GPU TRUE)
else()
    message(STATUS "CUDA NOT found! Compiling CPU fallback targets only.")
    set(HAS_GPU FALSE)
endif()

# 3. Handle OpenMP Configuration Natively across Linux and macOS
find_package(OpenMP REQUIRED)

# -----------------------------------------------
# Target 1: CPU Beamforming Engine
# -----------------------------------------------
python_add_library(beamform_cpu MODULE beampower/src/beamform.c)

target_compile_options(beamform_cpu PRIVATE 
    -O3 
    -fPIC 
    -ftree-vectorize 
    -march=native
)

# Connect OpenMP variables natively found by CMake (handles Homebrew/venvs/Linux automatically)
target_link_libraries(beamform_cpu PRIVATE OpenMP::OpenMP_C)

# Install into the output layout inside the destination module's lib directory
install(TARGETS beamform_cpu DESTINATION beampower/lib)

# -----------------------------------------------
# Target 2: GPU Beamforming Engine (Conditional)
# -----------------------------------------------
if(HAS_GPU)
    python_add_library(beamform_gpu MODULE beampower/src/beamform.cu)
    
    # Mirroring your old architectural flags (-gencode arch=compute_...)
    set_target_properties(beamform_gpu PROPERTIES
        CUDA_ARCHITECTURES "70;75;80"
    )
    
    # Pass host flags safely via CMake's specialized wrapper syntax
    target_compile_options(beamform_gpu PRIVATE
        -O3
        $<$<COMPILE_LANGUAGE:CUDA>:
            -Xcompiler "-fopenmp -fPIC -march=native -ftree-vectorize"
            -Xlinker -lgomp
        >
    )
    
    target_link_libraries(beamform_gpu PRIVATE OpenMP::OpenMP_C)
    install(TARGETS beamform_gpu DESTINATION beampower/lib)
endif()