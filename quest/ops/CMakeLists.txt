cmake_minimum_required(VERSION 3.26.4)

set(CMAKE_C_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

project(_kernels LANGUAGES CUDA CXX)

# Use generic Python find package which pybind11 seems to prefer or fallback to
set(Python_ROOT_DIR "/opt/dlami/nvme/mira/conda-envs/quest")
set(Python_FIND_STRATEGY LOCATION)
# Try to find Python 3 specifically but using the "Python" module name if possible
# or just "Python" which implies 3 in newer cmake
find_package(Python REQUIRED COMPONENTS Interpreter Development)

message(STATUS "Found Python: ${Python_EXECUTABLE}")
message(STATUS "Python Include: ${Python_INCLUDE_DIRS}")

# Get Torch path
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_PREFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
message(STATUS "Torch CMake prefix: ${TORCH_CMAKE_PREFIX}")
list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PREFIX}")

find_package(Torch REQUIRED)
find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")

# Pybind11
add_subdirectory(${CMAKE_SOURCE_DIR}/../../kernels/3rdparty/pybind ${CMAKE_BINARY_DIR}/pybind11)

file(GLOB PYTORCH_SOURCES "csrc/*.cu")

# Create the module
pybind11_add_module(_kernels MODULE ${PYTORCH_SOURCES})

target_compile_definitions(_kernels PRIVATE -DBSK_TORCH_CHECK)
target_include_directories(_kernels PRIVATE ${CMAKE_SOURCE_DIR}/../../kernels/include)
target_include_directories(_kernels PRIVATE ${CMAKE_SOURCE_DIR}/../../kernels/3rdparty/flashinfer/include)
target_include_directories(_kernels PRIVATE ${CMAKE_SOURCE_DIR}/../../kernels/3rdparty/pybind/include)

target_compile_options(_kernels PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda --expt-relaxed-constexpr>)

# Link libraries
target_link_libraries(_kernels PRIVATE ${TORCH_LIBRARIES} Python::Python ${TORCH_PYTHON_LIBRARY})
