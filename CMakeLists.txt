# Defina a versão mínima do CMake
cmake_minimum_required(VERSION 3.10)

# Nome do projeto
project(KDE_Project)

# Definir o padrão de C++ (como C++17)
set(CMAKE_CXX_STANDARD 17)

# Habilitar o suporte ao CUDA
find_package(CUDA REQUIRED)

# Buscar o OpenCV
find_package(OpenCV REQUIRED)

# Adicionar arquivos-fonte C++ e CUDA
set(SOURCES
    kde_cuda.cu  # Arquivo CUDA
)

# Incluir os diretórios de headers (se necessário)
include_directories(${CMAKE_SOURCE_DIR} ${OpenCV_INCLUDE_DIRS})

# Adicionar o executável que será gerado
cuda_add_executable(kde_program ${SOURCES})

# Linkar com a biblioteca OpenCV
target_link_libraries(kde_program ${OpenCV_LIBS})

# Configurações adicionais
set_target_properties(kde_program PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CXX_STANDARD 17
)
