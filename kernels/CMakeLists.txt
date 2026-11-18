cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

project(mixfp4 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Build only CUTLASS's headers + host/device reference utilities (tools/util) --
# skip its own examples/tests/profiler, which we don't need and which are slow to configure.
set(CUTLASS_ENABLE_HEADERS_ONLY ON CACHE BOOL "" FORCE)

add_subdirectory(3rdparty/cutlass)

# This repository now targets two different Blackwell tensor-core ISAs. They share almost no
# kernel code -- sm_120 (GeForce) issues per-warp `mma.sync...block_scale`, sm_100 (datacenter)
# issues CTA-wide `tcgen05.mma` -- so which set of targets gets built is a configure-time choice.
#
#   -DMIXFP4_ARCH=120a   GeForce Blackwell (RTX PRO 6000 / RTX 5090)   [default]
#   -DMIXFP4_ARCH=100a   Datacenter Blackwell (B200/GB200)
set(MIXFP4_ARCH "120a" CACHE STRING "Target tensor-core ISA: 120a (GeForce) or 100a (datacenter)")
set_property(CACHE MIXFP4_ARCH PROPERTY STRINGS 120a 100a)

if(MIXFP4_ARCH STREQUAL "120a")

# ---- ctypes-loadable C API (capi/README.md) ------------------------------------------------------
# libflipquant_sm120_unpatched.so is the compiled library; libflipquant_sm120.so is the same file
# after scripts/patch_mixed_nvfp4_gemm.py has installed the E0M3 SASS sites and flipped the
# library's patch marker. Build target `flipquant_sm120` produces the patched one; the unpatched
# library refuses to run.
#
# Kernel configuration: weights on CUTLASS operand A at a 16-row x 64-K granule, activations
# pinned to E2M1 (sm120/kernel/docs/mixed_nvfp4_report.md section 6, "WEIGHTS AS THE A OPERAND"). Its blob
# header is generated into the build tree and put first on the include path, so the checked-in
# src/collective/mixed_mma_blob_generated.hpp is not touched.
find_package(CUDAToolkit REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter)
find_package(OpenMP)
include(CheckCompilerFlag)
# The patcher edits the fatbin in place, so device code must not be compressed inside the .so.
check_compiler_flag(CUDA "--no-compress" FQ_NVCC_HAS_NO_COMPRESS)

set(FQ_SM120_GEN_DIR "${CMAKE_CURRENT_BINARY_DIR}/capi_gen_sm120")
set(FQ_SM120_BLOB "${FQ_SM120_GEN_DIR}/collective/mixed_mma_blob_generated.hpp")
add_custom_command(
  OUTPUT "${FQ_SM120_BLOB}"
  COMMAND ${CMAKE_COMMAND} -E env MMA_M=2 MMA_N=8 A_ATOMS=1 B_ATOMS=8
          "MIXFP4_BLOB_OUT=${FQ_SM120_BLOB}"
          "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/scripts/gen_mixed_mma_blob.py"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/scripts/gen_mixed_mma_blob.py"
  COMMENT "Generating the 16x64 weights-as-A MMA blob for the sm120 C API"
  VERBATIM)
add_custom_target(flipquant_sm120_blob DEPENDS "${FQ_SM120_BLOB}")

add_library(flipquant_sm120_unpatched SHARED capi/fq_sm120_capi.cu)
add_dependencies(flipquant_sm120_unpatched flipquant_sm120_blob)
cutlass_apply_standard_compile_options(flipquant_sm120_unpatched)
target_include_directories(flipquant_sm120_unpatched BEFORE PRIVATE "${FQ_SM120_GEN_DIR}")
target_include_directories(flipquant_sm120_unpatched PRIVATE src capi)
target_link_libraries(flipquant_sm120_unpatched PRIVATE nvidia::cutlass::tools::util CUDA::cuda_driver)
target_compile_definitions(flipquant_sm120_unpatched PRIVATE
  MIXFP4_BLOB=1 MIXFP4_JOINT_KA=1 MIXFP4_B_ALL_E2M1=1
  MIXFP4_A_ATOMS_PER_GRANULE=1 MIXFP4_B_ATOMS_PER_GRANULE=8)
# SASS only, no PTX: embedded PTX would be JIT-compiled UNPATCHED on any other device.
set_property(TARGET flipquant_sm120_unpatched PROPERTY CUDA_ARCHITECTURES 120a-real)
set_target_properties(flipquant_sm120_unpatched PROPERTIES
  CUDA_VISIBILITY_PRESET hidden CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)
target_compile_options(flipquant_sm120_unpatched PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
if(FQ_NVCC_HAS_NO_COMPRESS)
  target_compile_options(flipquant_sm120_unpatched PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--no-compress>)
endif()

set(FQ_SM120_SO "${CMAKE_CURRENT_BINARY_DIR}/libflipquant_sm120.so")
add_custom_command(
  OUTPUT "${FQ_SM120_SO}"
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/scripts/patch_mixed_nvfp4_gemm.py"
          "$<TARGET_FILE:flipquant_sm120_unpatched>" "${FQ_SM120_SO}"
          --cuobjdump "${CUDAToolkit_BIN_DIR}/cuobjdump"
          --expect-sites 0,1
          --mark-patched FQ_SM120_PATCH_STATE
  DEPENDS flipquant_sm120_unpatched "${CMAKE_CURRENT_SOURCE_DIR}/scripts/patch_mixed_nvfp4_gemm.py"
  COMMENT "Patching E0M3 SASS sites into libflipquant_sm120.so"
  VERBATIM)
add_custom_target(flipquant_sm120 ALL DEPENDS "${FQ_SM120_SO}")

add_executable(test_capi_sm120 capi/test_capi_sm120.cu)
target_include_directories(test_capi_sm120 PRIVATE capi)
target_link_libraries(test_capi_sm120 PRIVATE CUDA::cudart ${CMAKE_DL_LIBS})
set_property(TARGET test_capi_sm120 PROPERTY CUDA_ARCHITECTURES 120a-real)
if(OpenMP_CXX_FOUND)
  target_compile_options(test_capi_sm120 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${OpenMP_CXX_FLAGS}>)
  target_link_libraries(test_capi_sm120 PRIVATE OpenMP::OpenMP_CXX)
endif()
add_dependencies(test_capi_sm120 flipquant_sm120)

# Second sm120 C API: WEIGHTS ON CUTLASS OPERAND B at an 8-row x 64-K granule (1x8 warp
# arrangement, PERM_N=128, activations pinned to E2M1 on A; sm120/kernel/docs/mixed_nvfp4_report.md, "The
# 8-column x 64-K frontier"). Separate blob header, separate library, separate patch marker; the
# weights-as-A library above is unchanged.
set(FQ_SM120B8_GEN_DIR "${CMAKE_CURRENT_BINARY_DIR}/capi_gen_sm120_b8")
set(FQ_SM120B8_BLOB "${FQ_SM120B8_GEN_DIR}/collective/mixed_mma_blob_generated.hpp")
add_custom_command(
  OUTPUT "${FQ_SM120B8_BLOB}"
  COMMAND ${CMAKE_COMMAND} -E env MMA_M=8 MMA_N=2 A_ATOMS=8 B_ATOMS=1
          "MIXFP4_BLOB_OUT=${FQ_SM120B8_BLOB}"
          "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/scripts/gen_mixed_mma_blob.py"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/scripts/gen_mixed_mma_blob.py"
  COMMENT "Generating the 8x64 weights-on-B MMA blob for the sm120_b8 C API"
  VERBATIM)
add_custom_target(flipquant_sm120_b8_blob DEPENDS "${FQ_SM120B8_BLOB}")

add_library(flipquant_sm120_b8_unpatched SHARED capi/fq_sm120_b8_capi.cu)
add_dependencies(flipquant_sm120_b8_unpatched flipquant_sm120_b8_blob)
cutlass_apply_standard_compile_options(flipquant_sm120_b8_unpatched)
target_include_directories(flipquant_sm120_b8_unpatched BEFORE PRIVATE "${FQ_SM120B8_GEN_DIR}")
target_include_directories(flipquant_sm120_b8_unpatched PRIVATE src capi)
target_link_libraries(flipquant_sm120_b8_unpatched PRIVATE nvidia::cutlass::tools::util CUDA::cuda_driver)
target_compile_definitions(flipquant_sm120_b8_unpatched PRIVATE
  MIXFP4_BLOB=1 MIXFP4_JOINT_KB=1 MIXFP4_A_ALL_E2M1=1 MIXFP4_ATOM_M=1 MIXFP4_PERM_N=128
  MIXFP4_A_ATOMS_PER_GRANULE=8 MIXFP4_B_ATOMS_PER_GRANULE=1)
set_property(TARGET flipquant_sm120_b8_unpatched PROPERTY CUDA_ARCHITECTURES 120a-real)
set_target_properties(flipquant_sm120_b8_unpatched PROPERTIES
  CUDA_VISIBILITY_PRESET hidden CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)
target_compile_options(flipquant_sm120_b8_unpatched PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
if(FQ_NVCC_HAS_NO_COMPRESS)
  target_compile_options(flipquant_sm120_b8_unpatched PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--no-compress>)
endif()

# Sites: A is pinned to E2M1, so exactly {0: e2m1 x e2m1, 2: e2m1 x e0m3} must exist (an exact
# census, stricter than --allow-missing-sites).
set(FQ_SM120B8_SO "${CMAKE_CURRENT_BINARY_DIR}/libflipquant_sm120_b8.so")
add_custom_command(
  OUTPUT "${FQ_SM120B8_SO}"
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/scripts/patch_mixed_nvfp4_gemm.py"
          "$<TARGET_FILE:flipquant_sm120_b8_unpatched>" "${FQ_SM120B8_SO}"
          --cuobjdump "${CUDAToolkit_BIN_DIR}/cuobjdump"
          --expect-sites 0,2
          --mark-patched FQ_SM120B8_PATCH_STATE
  DEPENDS flipquant_sm120_b8_unpatched "${CMAKE_CURRENT_SOURCE_DIR}/scripts/patch_mixed_nvfp4_gemm.py"
  COMMENT "Patching E0M3 SASS sites into libflipquant_sm120_b8.so"
  VERBATIM)
add_custom_target(flipquant_sm120_b8 ALL DEPENDS "${FQ_SM120B8_SO}")

add_executable(test_capi_sm120_b8 capi/test_capi_sm120_b8.cu)
target_include_directories(test_capi_sm120_b8 PRIVATE capi)
target_link_libraries(test_capi_sm120_b8 PRIVATE CUDA::cudart ${CMAKE_DL_LIBS})
set_property(TARGET test_capi_sm120_b8 PROPERTY CUDA_ARCHITECTURES 120a-real)
if(OpenMP_CXX_FOUND)
  target_compile_options(test_capi_sm120_b8 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${OpenMP_CXX_FLAGS}>)
  target_link_libraries(test_capi_sm120_b8 PRIVATE OpenMP::OpenMP_CXX)
endif()
add_dependencies(test_capi_sm120_b8 flipquant_sm120_b8)

endif()  # MIXFP4_ARCH STREQUAL "120a"

if(MIXFP4_ARCH STREQUAL "100a")

find_package(CUDAToolkit REQUIRED)

# Datacenter Blackwell block-scaled NVFP4 through tcgen05.mma. Unlike sm_120a, the "a" here still
# means architecture-specific (no PTX forward compatibility), so this runs on sm_100 only.
add_executable(nvfp4_gemm_sm100 src/nvfp4_gemm_sm100.cu)
cutlass_apply_standard_compile_options(nvfp4_gemm_sm100)
target_link_libraries(nvfp4_gemm_sm100 PRIVATE nvidia::cutlass::tools::util CUDA::cuda_driver)
target_include_directories(nvfp4_gemm_sm100 PRIVATE 3rdparty/cutlass/examples/common src)
cutlass_apply_cuda_gencode_flags(nvfp4_gemm_sm100 SM_ARCHS 100a)
target_compile_options(nvfp4_gemm_sm100 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)


# Mixed E0M3/E2M1 through a runtime UMMA instruction descriptor. Needs OpenMP for the host-side
# E0M3 reference, which is a plain O(M*N*K) float GEMM.
find_package(OpenMP)
add_executable(mixed_nvfp4_gemm_sm100 src/mixed_nvfp4_gemm_sm100.cu)
cutlass_apply_standard_compile_options(mixed_nvfp4_gemm_sm100)
target_link_libraries(mixed_nvfp4_gemm_sm100 PRIVATE nvidia::cutlass::tools::util CUDA::cuda_driver)
target_include_directories(mixed_nvfp4_gemm_sm100 PRIVATE 3rdparty/cutlass/examples/common src)
cutlass_apply_cuda_gencode_flags(mixed_nvfp4_gemm_sm100 SM_ARCHS 100a)
target_compile_options(mixed_nvfp4_gemm_sm100 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
# --forward (per-forward GEMM timing, paper D.5) reads CUPTI kernel times when CUPTI is available.
function(mixfp4_link_cupti name)
  if(TARGET CUDA::cupti)
    target_link_libraries(${name} PRIVATE CUDA::cupti)
    target_compile_definitions(${name} PRIVATE MIXFP4_HAVE_CUPTI=1)
  endif()
endfunction()
mixfp4_link_cupti(mixed_nvfp4_gemm_sm100)
if(OpenMP_CXX_FOUND)
  target_compile_options(mixed_nvfp4_gemm_sm100 PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${OpenMP_CXX_FLAGS}>)
  target_link_libraries(mixed_nvfp4_gemm_sm100 PRIVATE OpenMP::OpenMP_CXX)
endif()

# MMA-width experiment (paper Table D.6): the tuned 256x256x256 2-SM kernel with each 256-wide MMA
# issued as 256/N narrower 2-SM MMAs, one B format per N x 64 tile. N=256 is the per-tile kernel.
function(mixfp4_sub_target name)
  add_executable(${name} src/mixed_nvfp4_gemm_sm100.cu)
  target_compile_definitions(${name} PRIVATE ${ARGN})
  cutlass_apply_standard_compile_options(${name})
  target_link_libraries(${name} PRIVATE nvidia::cutlass::tools::util CUDA::cuda_driver)
  target_include_directories(${name} PRIVATE 3rdparty/cutlass/examples/common src)
  cutlass_apply_cuda_gencode_flags(${name} SM_ARCHS 100a)
  target_compile_options(${name} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
  mixfp4_link_cupti(${name})
  if(OpenMP_CXX_FOUND)
    target_compile_options(${name} PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${OpenMP_CXX_FLAGS}>)
    target_link_libraries(${name} PRIVATE OpenMP::OpenMP_CXX)
  endif()
endfunction()
foreach(sub_n IN ITEMS 16 32 64 128 256)
  mixfp4_sub_target(mixed_sub_n${sub_n}_sm100 MIXFP4_SUB_N=${sub_n})
endforeach()
# 1-SM 128x256x256 tile: the only way to issue N=8.
foreach(sub_n IN ITEMS 8 16 32 64 128 256)
  mixfp4_sub_target(mixed_sub1sm_n${sub_n}_sm100 MIXFP4_SUB_N=${sub_n} MIXFP4_TILE_M=128 MIXFP4_CLUSTER_M=1)
endforeach()
# Padded B scale factors (2x in smem/TMA, one pipeline stage fewer): the layout N=8/16/32 need,
# since an MMA's scale base must be lane 0 of an even TMEM column. N >= 64 built the same way for a
# like-for-like row.
# mixed_subpad*: the builder's stage count when the extra SFB still fits; mixed_subpadlo*: one fewer.
foreach(sub_n IN ITEMS 16 32 64 128 256)
  mixfp4_sub_target(mixed_subpad_n${sub_n}_sm100 MIXFP4_SUB_N=${sub_n} MIXFP4_SFB_PAD=1)
  mixfp4_sub_target(mixed_subpadlo_n${sub_n}_sm100 MIXFP4_SUB_N=${sub_n} MIXFP4_SFB_PAD=1 MIXFP4_STAGE_DELTA=1)
endforeach()
foreach(sub_n IN ITEMS 8 16 32 64 128 256)
  mixfp4_sub_target(mixed_subpad1sm_n${sub_n}_sm100 MIXFP4_SUB_N=${sub_n} MIXFP4_SFB_PAD=1
                    MIXFP4_TILE_M=128 MIXFP4_CLUSTER_M=1)
  mixfp4_sub_target(mixed_subpadlo1sm_n${sub_n}_sm100 MIXFP4_SUB_N=${sub_n} MIXFP4_SFB_PAD=1 MIXFP4_STAGE_DELTA=1
                    MIXFP4_TILE_M=128 MIXFP4_CLUSTER_M=1)
endforeach()
# Hardware probes for the scale-factor TMEM layout (wrong numerics by design; they only ask which
# SFB MMA bases and copy offsets the hardware accepts). See scripts/probe_sm100_sfb.sh.
mixfp4_sub_target(probe_n32_sfb2_sm100 MIXFP4_SUB_N=32 MIXFP4_DIAG_SFB_ALIGN=2)
mixfp4_sub_target(probe_n32_sfb4_sm100 MIXFP4_SUB_N=32 MIXFP4_DIAG_SFB_ALIGN=4)
mixfp4_sub_target(probe_n16_sfb2_sm100 MIXFP4_SUB_N=16 MIXFP4_DIAG_SFB_ALIGN=2)
mixfp4_sub_target(probe_n16_sfb2_lane_sm100 MIXFP4_SUB_N=16 MIXFP4_DIAG_SFB_ALIGN=2 MIXFP4_DIAG_SFB_LANE=1)
mixfp4_sub_target(probe_cp_dst1_sm100 MIXFP4_SUB_N=256 MIXFP4_DIAG_CP_DST=1)
mixfp4_sub_target(probe_cp_dst2_sm100 MIXFP4_SUB_N=256 MIXFP4_DIAG_CP_DST=2)
mixfp4_sub_target(probe_cp_src16_sm100 MIXFP4_SUB_N=256 MIXFP4_DIAG_CP_SRC=16)
mixfp4_sub_target(probe1sm_n8_sfb2_sm100 MIXFP4_SUB_N=8 MIXFP4_TILE_M=128 MIXFP4_CLUSTER_M=1 MIXFP4_DIAG_SFB_ALIGN=2)
mixfp4_sub_target(probe1sm_n8_sfb4_sm100 MIXFP4_SUB_N=8 MIXFP4_TILE_M=128 MIXFP4_CLUSTER_M=1 MIXFP4_DIAG_SFB_ALIGN=4)
mixfp4_sub_target(probe1sm_n8_sfb4_lane_sm100 MIXFP4_SUB_N=8 MIXFP4_TILE_M=128 MIXFP4_CLUSTER_M=1 MIXFP4_DIAG_SFB_ALIGN=4 MIXFP4_DIAG_SFB_LANE=1)

# Standalone N8 x K64 kernel: the correctness driver of the pipeline behind libflipquant_sm100.so
# (scripts/check_sm100_n8_pipeline.sh).
add_executable(mixed_n8k64_sm100 src/mixed_n8k64_sm100.cu)

target_link_libraries(mixed_n8k64_sm100 PRIVATE nvidia::cutlass::tools::util)
target_include_directories(mixed_n8k64_sm100 PRIVATE src)
cutlass_apply_standard_compile_options(mixed_n8k64_sm100)
cutlass_apply_cuda_gencode_flags(mixed_n8k64_sm100 SM_ARCHS 100a)
if(OpenMP_CXX_FOUND)
  target_compile_options(mixed_n8k64_sm100 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${OpenMP_CXX_FLAGS}>)
  target_link_libraries(mixed_n8k64_sm100 PRIVATE OpenMP::OpenMP_CXX)
endif()

# ---- ctypes-loadable C API (capi/README.md) ------------------------------------------------------
# Exact per-(N8 x K64) weight formats via the validated packed-input N8 pipeline
# (src/n8k64_pipeline.cuh, CTA 128x32x128). No SASS patching on sm_100: the format is a field of
# the runtime UMMA instruction descriptor.
add_library(flipquant_sm100 SHARED capi/fq_sm100_capi.cu)
cutlass_apply_standard_compile_options(flipquant_sm100)
target_include_directories(flipquant_sm100 PRIVATE src capi)
target_link_libraries(flipquant_sm100 PRIVATE nvidia::cutlass::tools::util)
set_property(TARGET flipquant_sm100 PROPERTY CUDA_ARCHITECTURES 100a-real)
set_target_properties(flipquant_sm100 PROPERTIES
  CUDA_VISIBILITY_PRESET hidden CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)
target_compile_options(flipquant_sm100 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)

# Native-inference C ABI (fq_sm100_linear.cu): the tuned stock and 256x64 per-tile kernels with the
# deployed epilogue, one .so; the source is compiled once per variant.
foreach(variant IN ITEMS stock tile)
  add_library(fq_sm100_linear_${variant} OBJECT src/fq_sm100_linear.cu)
  cutlass_apply_standard_compile_options(fq_sm100_linear_${variant})
  target_include_directories(fq_sm100_linear_${variant} PRIVATE src sm120/csrc)
  target_link_libraries(fq_sm100_linear_${variant} PRIVATE nvidia::cutlass::tools::util CUDA::cuda_driver)
  set_property(TARGET fq_sm100_linear_${variant} PROPERTY CUDA_ARCHITECTURES 100a-real)
  set_property(TARGET fq_sm100_linear_${variant} PROPERTY POSITION_INDEPENDENT_CODE ON)
  set_target_properties(fq_sm100_linear_${variant} PROPERTIES
    CUDA_VISIBILITY_PRESET hidden CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)
  target_compile_options(fq_sm100_linear_${variant} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
endforeach()
target_compile_definitions(fq_sm100_linear_tile PRIVATE FQ_SM100_TILE=1)
add_library(flipquant_sm100_linear SHARED $<TARGET_OBJECTS:fq_sm100_linear_stock>
                                          $<TARGET_OBJECTS:fq_sm100_linear_tile>)
target_link_libraries(flipquant_sm100_linear PRIVATE CUDA::cudart CUDA::cuda_driver)
set_property(TARGET flipquant_sm100_linear PROPERTY CUDA_ARCHITECTURES 100a-real)

add_executable(test_capi_sm100 capi/test_capi_sm100.cu)
target_include_directories(test_capi_sm100 PRIVATE capi)
target_link_libraries(test_capi_sm100 PRIVATE CUDA::cudart ${CMAKE_DL_LIBS})
set_property(TARGET test_capi_sm100 PROPERTY CUDA_ARCHITECTURES 100a-real)
if(OpenMP_CXX_FOUND)
  target_compile_options(test_capi_sm100 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${OpenMP_CXX_FLAGS}>)
  target_link_libraries(test_capi_sm100 PRIVATE OpenMP::OpenMP_CXX)
endif()
add_dependencies(test_capi_sm100 flipquant_sm100)

endif()  # MIXFP4_ARCH STREQUAL "100a"
