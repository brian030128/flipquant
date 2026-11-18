// Experimental fine-grained SM100 GEMM. Each weight-format byte describes N8 x K64.
// Includes the synchronous baseline and an opt-in double-buffered async path.
// Do not interpret either full GEMM speed as the hardware ceiling of N8 MMA.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <random>
#include <string>
#include <vector>
#include "cute/tensor.hpp"
#include "cute/arch/mma_sm100_umma.hpp"
#include "cute/arch/tmem_allocator_sm100.hpp"
#include "cutlass/arch/barrier.h"
#include "cutlass/bfloat16.h"
#include "gemm_benchmark.hpp"

using namespace cute;
using F4 = cutlass::float_e2m1_t;
using SF = cutlass::float_ue4m3_t;
using BF16 = cutlass::bfloat16_t;
#include "n8k64_pipeline.cuh"

// Packed inputs are row-major A[M,K] and B[N,K]. Scales here are ordinary
// row-major [M or N,K/16], not the shuffled scale layout of the CUTLASS driver.
// Formats are row-major [N/8,K/64]; activation format is uniform for this probe.
template<int TileN, int AtomN, int TileK>
__global__ void n8k64_gemm(uint8_t const* a, uint8_t const* b,
                         uint8_t const* sa, uint8_t const* sb,
                         uint8_t const* formats, BF16* d,
                         int m, int n, int k, int fmt_a, int resident_repeats) {
  static_assert(TileN % AtomN == 0 && AtomN % 8 == 0);
  constexpr int Groups = TileN / AtomN;
  constexpr int KGroups = TileK / 64;
  // MMA scale base addresses need four-column alignment, even for N=8; an
  // N-wide atom stores its weight scales in N/32 columns (8 at N=256).
  constexpr int BCols = AtomN / 32 > 4 ? AtomN / 32 : 4;
  constexpr int Required = TileN + KGroups * (4 + Groups * BCols);
  constexpr int TmemCols = Required <= 32 ? 32 : Required <= 64 ? 64 :
                           Required <= 128 ? 128 : Required <= 256 ? 256 : 512;
  static_assert(Required <= 512);
  __shared__ __align__(128) uint8_t as[128 * TileK / 2];
  __shared__ __align__(128) uint8_t bs[TileN * TileK / 2];
  __shared__ uint32_t tm;
  __shared__ __align__(16) uint64_t barrier;
  int tid = int(threadIdx.x), lane = tid % 32, warp = tid / 32;
  int row = int(blockIdx.x) * 128;
  int col = int(blockIdx.y) * TileN;
  TMEM::Allocator1Sm allocator;
  if (warp == 0) allocator.allocate(TmemCols, &tm);
  if (tid == 0) initialize_barrier(barrier, 1);
  __syncthreads();
  if (warp == 0) allocator.release_allocation_lock();

  // Interleaved SMEM: 16-byte K chunks, then rows, then the second K chunk.
  auto descriptor = [](uint8_t* ptr, int rows) {
    UMMA::SmemDescriptor desc{};
    desc.version_ = 1;
    desc.start_address_ = uint32_t(__cvta_generic_to_shared(ptr)) >> 4;
    desc.leading_byte_offset_ = rows;
    desc.stride_byte_offset_ = 8;
    return desc;
  };
  auto idesc = UMMA::make_instr_desc_block_scaled<F4,F4,float,SF,128,AtomN,
                                                  UMMA::Major::K,UMMA::Major::K>();
  idesc.a_format_ = uint8_t(fmt_a);
  int phase = 0;
  for (int kk = 0; kk < k; kk += TileK) {
    for (int chunk = tid; chunk < 128 * TileK / 32; chunk += 128) {
      int r = chunk % 128, half = chunk / 128;
      uint4 value{};
      if (row + r < m && kk + half*32 < k) value = *reinterpret_cast<uint4 const*>(
          a + size_t(row+r)*(k/2) + kk/2 + half*16);
      reinterpret_cast<uint4*>(as)[chunk] = value;
    }
    for (int chunk = tid; chunk < TileN * TileK / 32; chunk += 128) {
      int c = chunk % TileN, half = chunk / TileN;
      uint4 value{};
      if (col+c < n && kk + half*32 < k) value = *reinterpret_cast<uint4 const*>(
          b + size_t(col+c)*(k/2) + kk/2 + half*16);
      reinterpret_cast<uint4*>(bs)[chunk] = value;
    }
    // Scale factors are replicated across all four 32-lane subpartitions.
    #pragma unroll
    for (int kg = 0; kg < KGroups; ++kg) {
      #pragma unroll
      for (int q = 0; q < 4; ++q) {
        int r = row + q*32 + lane;
        uint32_t value = r < m && kk+kg*64 < k ? *reinterpret_cast<uint32_t const*>(
            sa + size_t(r)*(k/16) + kk/16 + kg*4) : 0;
        SM100_TMEM_STORE_32dp32b1x::copy(value, tm + TileN + kg*4 + q + (warp*32u << 16));
      }
      #pragma unroll
      for (int g = 0; g < Groups; ++g) {
        #pragma unroll
        for (int q = 0; q < (AtomN+31)/32; ++q) {
          int c = col + g*AtomN + q*32 + lane;
          uint32_t value = c < n && q*32+lane < AtomN && kk+kg*64 < k ?
              *reinterpret_cast<uint32_t const*>(sb + size_t(c)*(k/16) + kk/16 + kg*4) : 0;
          SM100_TMEM_STORE_32dp32b1x::copy(value,
              tm + TileN + KGroups*4 + (kg*Groups+g)*BCols + q + (warp*32u << 16));
        }
      }
    }
    cutlass::arch::fence_view_async_tmem_store();
    cutlass::arch::fence_view_async_shared();
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
    if (warp == 0) {
      // Diagnostic mode repeats resident operands, excluding repeated global
      // data/scale loads. It is an MMA issue test, NOT a full GEMM benchmark.
      for (int rep = 0; rep < resident_repeats; ++rep) {
        #pragma unroll
        for (int kg = 0; kg < KGroups; ++kg) {
          if (kk+kg*64 >= k) break;
          auto da = descriptor(as + kg*128*32, 128);
          #pragma unroll
          for (int g = 0; g < Groups; ++g) {
            auto db = descriptor(bs + kg*TileN*32 + g*AtomN*16, TileN);
            int c = col + g*AtomN;
            idesc.b_format_ = c < n ? formats[size_t(c/8)*(k/64) + kk/64 + kg] : 1;
            uint32_t tsfa = tm + TileN + kg*4;
            uint32_t tsfb = tm + TileN + KGroups*4 + (kg*Groups+g)*BCols;
            auto desc = UMMA::make_runtime_instr_desc_block_scaled(idesc, tsfa, tsfb);
            SM100_MMA_MXF4_SS<F4,F4,float,SF,128,AtomN,16,
                UMMA::Major::K,UMMA::Major::K>::fma(da, db, tm + g*AtomN,
                    kk+kg*64 != 0 || rep != 0, desc, tsfa, tsfb);
          }
        }
      }
      cutlass::arch::umma_arrive(&barrier);
    }
    wait_barrier(barrier, phase);
    phase ^= 1;
    __syncthreads();
  }
  #pragma unroll
  for (int c = 0; c < TileN; c += 8) {
    uint32_t v[8];
    SM100_TMEM_LOAD_32dp32b8x::copy(tm + c + (warp*32u << 16),
        v[0],v[1],v[2],v[3],v[4],v[5],v[6],v[7]);
    cutlass::arch::fence_view_async_tmem_load();
    #pragma unroll
    for (int j = 0; j < 8; ++j)
      if (row+tid < m && col+c+j < n)
        d[size_t(row+tid)*n + col+c+j] = BF16(__uint_as_float(v[j]));
  }
  __syncthreads();
  if (warp == 0) allocator.free(tm, TmemCols);
}

template<class T> T* upload(std::vector<T> const& host) {
  T* device;
  mixfp4::benchmark_cuda_check(cudaMalloc(&device, host.size()*sizeof(T)));
  mixfp4::benchmark_cuda_check(cudaMemcpy(device, host.data(), host.size()*sizeof(T), cudaMemcpyHostToDevice));
  return device;
}

int main(int argc, char** argv) {
  int m=256,n=128,k=256, tile=64, tile_k=256, atom=8, iters=20, fa=1, resident_repeats=1;
  bool check=true;
  bool pipeline=false;
  bool packed_inputs=false;
  bool tile_n_set=false, tile_k_set=false;
  std::string pattern="checker";
  if (argc >= 4) { m=std::atoi(argv[1]); n=std::atoi(argv[2]); k=std::atoi(argv[3]); }
  for (int i=4;i<argc;++i) {
    std::string arg=argv[i];
    if (arg.rfind("--tile-n=",0)==0) { tile=std::stoi(arg.substr(9)); tile_n_set=true; }
    else if (arg.rfind("--tile-k=",0)==0) { tile_k=std::stoi(arg.substr(9)); tile_k_set=true; }
    else if (arg.rfind("--atom-n=",0)==0) atom=std::stoi(arg.substr(9));
    else if (arg.rfind("--iters=",0)==0) iters=std::stoi(arg.substr(8));
    else if (arg.rfind("--resident-repeats=",0)==0) resident_repeats=std::stoi(arg.substr(19));
    else if (arg.rfind("--pattern=",0)==0) pattern=arg.substr(10);
    else if (arg=="--fmt-a=e0m3") fa=0;
    else if (arg=="--fmt-a=e2m1") fa=1;
    else if (arg=="--no-check") check=false;
    else if (arg=="--pipeline") pipeline=true;
    else if (arg=="--packed-inputs") { pipeline=true; packed_inputs=true; }
    else { std::fprintf(stderr,"Unknown option: %s\n",argv[i]); return 2; }
  }
  if(pipeline && !tile_n_set) tile=32;
  if(pipeline && !tile_k_set) tile_k=128;
  // Width sweep: CTA 128x256x64 for every MMA width, so only the MMA N varies.
  bool sweep=tile==256;
  if (m<=0 || n<=0 || k<=0 || n%8 || k%64 || iters<=0 ||
      (tile!=8 && tile!=64 && tile!=128 && !sweep && !(pipeline && (tile==16 || tile==32))) ||
      (sweep ? (pipeline || tile_k!=64 || (atom!=8 && atom!=16 && atom!=32 && atom!=64 &&
                atom!=128 && atom!=256)) : atom!=8 && atom!=tile) ||
      (tile_k!=64 && tile_k!=256 && !(pipeline && tile_k==128)) ||
      (pipeline && (tile==8 || atom!=8 || resident_repeats!=1 || ((tile==16 || tile==128) && tile_k==256))) ||
      resident_repeats<1 || (resident_repeats!=1 && (check || k!=64 || tile_k!=64)) ||
      (pattern!="checker" && pattern!="random" && pattern!="e2m1" && pattern!="e0m3") ||
      (atom!=8 && !sweep && pattern!="e2m1" && pattern!="e0m3")) {
    std::fprintf(stderr,"Require N%%8=0, K%%64=0. Baseline: tile-n=8/64/128, tile-k=64/256. Width sweep: tile-n=256, tile-k=64, atom-n=8/16/32/64/128/256. Pipeline: tile-n=16/32/64/128, tile-k=64/128/256 (K256 only for N32/64), atom-n=8. Resident repeats require the baseline, --no-check, K=64 and tile-k=64.\n");
    return 2;
  }
  cudaDeviceProp props{};
  mixfp4::benchmark_cuda_check(cudaGetDeviceProperties(&props,0));
  if (props.major!=10) { std::fprintf(stderr,"Requires SM100.\n"); return 2; }
  std::mt19937 rng(2026);
  std::vector<uint8_t> a(size_t(m)*k/2), b(size_t(n)*k/2),
      sa(size_t(m)*k/16), sb(size_t(n)*k/16), formats(size_t(n/8)*(k/64));
  for (auto& v:a) v=check ? uint8_t(rng()) : 0x22;
  for (auto& v:b) v=check ? uint8_t(rng()) : 0x22;
  for (auto& v:sa) v=check ? uint8_t(0x30 + rng()%16) : 0x38;
  for (auto& v:sb) v=check ? uint8_t(0x30 + rng()%16) : 0x38;
  // One format per atom: an AtomN-wide MMA reads the entry of its first N8 group,
  // so every N8 group of an atom repeats it and the N8 reference stays exact.
  int per_atom=atom/8;
  for (int nn=0;nn<n/8;++nn) for (int kk=0;kk<k/64;++kk)
    formats[size_t(nn)*(k/64)+kk] = nn%per_atom ? formats[size_t(nn-nn%per_atom)*(k/64)+kk] :
        pattern=="e2m1" ? 1 : pattern=="e0m3" ? 0 :
        pattern=="random" ? uint8_t(rng()%2) : uint8_t((nn/per_atom+kk)%2);
  auto ad=upload(a), bd=upload(b), sad=upload(sa), sbd=upload(sb), fd=upload(formats);
  uint32_t *spa=nullptr, *spb=nullptr;
  uint8_t *pa=nullptr, *pb=nullptr;
  double pack_ms=0;
  std::function<void(cudaStream_t)> pack;
  if(pipeline) {
    mixfp4::benchmark_cuda_check(cudaMalloc(&spa,sa.size()));
    mixfp4::benchmark_cuda_check(cudaMalloc(&spb,sb.size()));
    if(packed_inputs) {
      mixfp4::benchmark_cuda_check(cudaMalloc(&pa,a.size()));
      mixfp4::benchmark_cuda_check(cudaMalloc(&pb,b.size()));
    }
    pack=[&](cudaStream_t stream) {
      n8_pack_scales<<<dim3((k/64+31)/32,(m+31)/32),dim3(32,8),0,stream>>>(
          reinterpret_cast<uint32_t const*>(sad),spa,m,k/64);
      n8_pack_scales<<<dim3((k/64+31)/32,(n+31)/32),dim3(32,8),0,stream>>>(
          reinterpret_cast<uint32_t const*>(sbd),spb,n,k/64);
      if(packed_inputs) {
        n8_pack_operands<<<dim3((k/32+31)/32,(m+31)/32),dim3(32,8),0,stream>>>(
            reinterpret_cast<uint4 const*>(ad),reinterpret_cast<uint4*>(pa),m,k/32);
        n8_pack_operands<<<dim3((k/32+31)/32,(n+31)/32),dim3(32,8),0,stream>>>(
            reinterpret_cast<uint4 const*>(bd),reinterpret_cast<uint4*>(pb),n,k/32);
      }
      mixfp4::benchmark_cuda_check(cudaGetLastError());
    };
    pack_ms=mixfp4::benchmark_ms(pack,2,iters);
  }
  BF16* dd;
  mixfp4::benchmark_cuda_check(cudaMalloc(&dd,size_t(m)*n*sizeof(BF16)));
  auto launch=[&](cudaStream_t stream) {
    dim3 grid((m+127)/128,(n+tile-1)/tile);
    if(pipeline) {
      #define PIPE(N,K) if(packed_inputs) { n8k64_pipeline<N,K,true><<<grid,128,0,stream>>>(pa,pb,spa,spb,fd,dd,m,n,k,fa); } else { n8k64_pipeline<N,K><<<grid,128,0,stream>>>(ad,bd,spa,spb,fd,dd,m,n,k,fa); }
      if(tile==16 && tile_k==64) { PIPE(16,64); }
      else if(tile==16) { PIPE(16,128); }
      else if(tile==32 && tile_k==64) { PIPE(32,64); }
      else if(tile==32 && tile_k==128) { PIPE(32,128); }
      else if(tile==32) { PIPE(32,256); }
      else if(tile==64 && tile_k==64) { PIPE(64,64); }
      else if(tile==64 && tile_k==128) { PIPE(64,128); }
      else if(tile==64) { PIPE(64,256); }
      else if(tile_k==64) { PIPE(128,64); }
      else { PIPE(128,128); }
      #undef PIPE
      mixfp4::benchmark_cuda_check(cudaGetLastError());
      return;
    }
    #define CALL(T,A,K) n8k64_gemm<T,A,K><<<grid,128,0,stream>>>(ad,bd,sad,sbd,fd,dd,m,n,k,fa,resident_repeats)
    #define LAUNCH(T,A) if(tile_k==64) { CALL(T,A,64); } else { CALL(T,A,256); }
    if (tile==8) { LAUNCH(8,8); }
    else if (tile==64 && atom==8) { LAUNCH(64,8); }
    else if (tile==64) { LAUNCH(64,64); }
    else if (tile==128 && atom==8) { LAUNCH(128,8); }
    else if (tile==128) { LAUNCH(128,128); }
    // TileK=256 would exceed TMEM (784 columns at N8) and static SMEM at TileN=256.
    else if (atom==8) { CALL(256,8,64); }
    else if (atom==16) { CALL(256,16,64); }
    else if (atom==32) { CALL(256,32,64); }
    else if (atom==64) { CALL(256,64,64); }
    else if (atom==128) { CALL(256,128,64); }
    else { CALL(256,256,64); }
    #undef LAUNCH
    #undef CALL
    mixfp4::benchmark_cuda_check(cudaGetLastError());
  };
  launch(nullptr);
  mixfp4::benchmark_cuda_check(cudaDeviceSynchronize());
  double ms=mixfp4::benchmark_ms(launch,2,iters);
  double end_to_end_ms=pipeline ? mixfp4::benchmark_ms([&](cudaStream_t stream) {
      pack(stream); launch(stream);
    },2,iters) : ms;
  bool pass=true;
  if (check) {
    std::vector<BF16> out(size_t(m)*n);
    mixfp4::benchmark_cuda_check(cudaMemcpy(out.data(),dd,out.size()*sizeof(BF16),cudaMemcpyDeviceToHost));
    auto decode=[](uint8_t packed,int idx,int fmt) {
      int nib=(packed >> ((idx%2)*4))&15;
      constexpr float e2[8]={0,.5,1,1.5,2,3,4,6};
      float v=fmt ? e2[nib&7] : float(nib&7);
      return nib&8 ? -v : v;
    };
    auto scale=[](uint8_t raw) { SF s; s.raw()=raw; return float(s); };
    double err=0, norm=0;
    int bad=0;
    #pragma omp parallel for reduction(+:err,norm,bad)
    for (int mm=0;mm<m;++mm) for (int nn=0;nn<n;++nn) {
      float ref=0;
      for (int kk=0;kk<k;++kk) {
        int fb=formats[size_t(nn/8)*(k/64)+kk/64];
        ref+=decode(a[size_t(mm)*(k/2)+kk/2],kk,fa)*scale(sa[size_t(mm)*(k/16)+kk/16])*
             decode(b[size_t(nn)*(k/2)+kk/2],kk,fb)*scale(sb[size_t(nn)*(k/16)+kk/16]);
      }
      float expected=float(BF16(ref)), actual=float(out[size_t(mm)*n+nn]);
      double diff=double(actual)-ref;
      err+=diff*diff; norm+=double(ref)*ref;
      if (!std::isfinite(actual) || actual!=expected) ++bad;
    }
    pass=bad==0;
    std::printf("Correctness: %s; mismatches=%d; relative L2=%.7f\n",pass?"PASSED":"FAILED",bad,std::sqrt(err/std::max(norm,1e-30)));
  }
  std::printf("Shape: %dx%dx%d; CTA: 128x%dx%d; MMA: 128x%dx64; weight formats: N%dxK64 (%s); A=%s\n",
      m,n,k,tile,tile_k,atom,atom,pattern.c_str(),fa?"e2m1":"e0m3");
  std::printf("Timing: %s; mode=%s; resident repeats=%d; latency %.6f ms; Throughput: %.1f TFLOP/s\n",
      mixfp4::benchmark_uses_graph()?"graph":"launch",resident_repeats==1?"gemm":"resident-mma",
      resident_repeats,ms,2.0*m*n*k*resident_repeats/ms/1e9);
  if(pipeline) {
    std::printf("Pipeline: double-buffered; packed inputs=%s; packing %.6f ms; packing + GEMM %.6f ms (%.1f TFLOP/s)\n",
        packed_inputs?"yes":"no",pack_ms,end_to_end_ms,2.0*m*n*k/end_to_end_ms/1e9);
    mixfp4::benchmark_cuda_check(cudaFree(spa));
    mixfp4::benchmark_cuda_check(cudaFree(spb));
    if(packed_inputs) {
      mixfp4::benchmark_cuda_check(cudaFree(pa));
      mixfp4::benchmark_cuda_check(cudaFree(pb));
    }
  }
  for (void* ptr:{static_cast<void*>(ad),static_cast<void*>(bd),static_cast<void*>(sad),
                 static_cast<void*>(sbd),static_cast<void*>(fd),static_cast<void*>(dd)})
    mixfp4::benchmark_cuda_check(cudaFree(ptr));
  return pass?0:1;
}
