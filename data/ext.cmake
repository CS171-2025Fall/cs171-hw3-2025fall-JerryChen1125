# All packages required
#
# Note that all the source code is downloaded from .zip files from GitHub.
# This is to avoid heavy traffic on Gitwhich may cause build failures.
#
# You can also use github mirrors to accelerate the download speed if needed.
 
# fmt (Tag: 9.1.0)
CPMAddPackage(
  NAME fmt
  URL https://github.com/fmtlib/fmt/archive/refs/tags/9.1.0.zip
)
 
# nlohmann_json (Tag: v3.11.2)
CPMAddPackage(
  NAME nlohmann_json
  URL https://github.com/nlohmann/json/archive/refs/tags/v3.11.2.zip
)
 
# tinyobjloader (Tag: v2.0.0rc10)
CPMAddPackage(
  NAME tinyobjloader
  URL https://github.com/tinyobjloader/tinyobjloader/archive/refs/tags/v2.0.0rc10.zip
)
 
# miniz (Tag: 3.0.2)
CPMAddPackage(
  NAME miniz
  URL https://github.com/richgel999/miniz/archive/refs/tags/3.0.2.zip
)
 
# spdlog (Tag: v1.11.0)
CPMAddPackage(
  NAME spdlog
  URL https://github.com/gabime/spdlog/archive/refs/tags/v1.11.0.zip
  OPTIONS "SPDLOG_FMT_EXTERNAL ON"
)
 
# tinyexr (Tag: v1.0.2)
CPMAddPackage(
  NAME tinyexr
  URL https://github.com/syoyo/tinyexr/archive/refs/tags/v1.0.2.zip
  DOWNLOAD_ONLY YES
)
 
if (tinyexr_ADDED)
    add_library(tinyexr STATIC ${tinyexr_SOURCE_DIR}/tinyexr.cc)
    target_include_directories(tinyexr INTERFACE ${tinyexr_SOURCE_DIR})
    target_compile_definitions(tinyexr PUBLIC -DTINYEXR_USE_MINIZ=1 -DTINYEXR_USE_PIZ=1
                                            -DTINYEXR_USE_OPENMP=0 -DTINYEXR_USE_STB_ZLIB=0)
    target_link_libraries(tinyexr PRIVATE miniz)
endif()
 
# linalg (Branch: main)
CPMAddPackage(
  NAME linalg
  URL https://github.com/sgorsten/linalg/archive/refs/heads/main.zip
  DOWNLOAD_ONLY YES
)
 
if (linalg_ADDED)
    add_library(linalg INTERFACE)
    target_include_directories(linalg INTERFACE ${linalg_SOURCE_DIR})
endif()
 
# stb (Branch: master)
CPMAddPackage(
  NAME stb
  URL https://github.com/nothings/stb/archive/refs/heads/master.zip
  DOWNLOAD_ONLY YES
)
 
if (stb_ADDED)
    add_library(stb INTERFACE)
    target_include_directories(stb INTERFACE ${stb_SOURCE_DIR})
    message(STATUS ${stb_SOURCE_DIR})
endif()
 
# googletest (Branch: main)
CPMAddPackage(
  NAME googletest
  URL https://github.com/google/googletest/archive/refs/heads/main.zip
  OPTIONS "INSTALL_GTEST OFF" "gtest_force_shared_crt"
)
 
if (USE_EMBREE)
    # External BVH library (Tag: v4.1.0)
    CPMAddPackage(
    NAME embree
    URL https://github.com/embree/embree/archive/refs/tags/v4.1.0.zip
    OPTIONS "EMBREE_ISPC_SUPPORT OFF"
            "EMBREE_TUTORIALS OFF"
            "EMBREE_FILTER_FUNCTION OFF"
            "EMBREE_RAY_PACKETS OFF"
            "EMBREE_RAY_MASK OFF"
            "EMBREE_GEOMETRY_GRID OFF"
            "EMBREE_GEOMETRY_QUAD OFF"
            "EMBREE_GEOMETRY_CURVE OFF"
            "EMBREE_GEOMETRY_SUBDIVISION OFF"
            "EMBREE_GEOMETRY_USER OFF"
            "EMBREE_GEOMETRY_POINT OFF"
            "EMBREE_DISC_POINT_SELF_INTERSE OFF"
 
            "EMBREE_MAX_ISA NONE"
            "EMBREE_ISA_AVX OFF"
            "EMBREE_ISA_AVX2 ON"
            "EMBREE_ISA_AVX512 OFF"
            "EMBREE_ISA_SSE2 OFF"
            "EMBREE_ISA_SSE42 OFF"
 
            "EMBREE_TASKING_SYSTEM INTERNAL")
endif()
 
