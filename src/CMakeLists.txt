find_package(OpenMP REQUIRED)

option(USE_CUDA "Enable CUDA Karras BVH builder" OFF)

# Gather sources
aux_source_directory("${PROJECT_SOURCE_DIR}/src" source)
list(REMOVE_ITEM source "${PROJECT_SOURCE_DIR}/src/main.cpp")
list(REMOVE_ITEM source "${PROJECT_SOURCE_DIR}/src/conventions.cpp")

if (USE_CUDA)
    list(APPEND source "${PROJECT_SOURCE_DIR}/src/karras_bvh_gpu.cu")
endif()

# Library
add_library(renderer_lib STATIC ${source})
add_library(renderer::lib ALIAS renderer_lib)

target_include_directories(renderer_lib PUBLIC "${PROJECT_SOURCE_DIR}/include")

target_link_libraries(renderer_lib PUBLIC
    stb
    OpenMP::OpenMP_CXX
    fmt
    nlohmann_json
    tinyobjloader
    linalg
    spdlog
    tinyexr
)

if (USE_EMBREE)
    target_link_libraries(renderer_lib PUBLIC embree)
    target_compile_definitions(renderer_lib PUBLIC USE_EMBREE)
endif()

# Executables
add_executable(renderer "${PROJECT_SOURCE_DIR}/src/main.cpp")
target_link_libraries(renderer PRIVATE renderer::lib)

add_executable(conventions "${PROJECT_SOURCE_DIR}/src/conventions.cpp")
target_link_libraries(conventions PRIVATE renderer::lib)

add_executable(exrtools "${PROJECT_SOURCE_DIR}/src/exrtools.cpp")
target_link_libraries(exrtools PRIVATE renderer::lib)