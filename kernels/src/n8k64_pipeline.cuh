#pragma once

// Transpose four adjacent scale bytes as one word. Both the source reads and
// destination writes are coalesced. Cost is reported separately by the driver.
__global__ void n8_pack_scales(uint32_t const* src, uint32_t* dst, int rows, int cols) {
  __shared__ uint32_t tile[32][33];
  int x = int(blockIdx.x)*32 + int(threadIdx.x);
  int y = int(blockIdx.y)*32 + int(threadIdx.y);
  #pragma unroll
  for (int j=0;j<32;j+=8)
    tile[threadIdx.y+j][threadIdx.x] = x<cols && y+j<rows ? src[size_t(y+j)*cols+x] : 0;
  __syncthreads();
  x = int(blockIdx.y)*32 + int(threadIdx.x);
  y = int(blockIdx.x)*32 + int(threadIdx.y);
  #pragma unroll
  for (int j=0;j<32;j+=8)
    if (x<rows && y+j<cols) dst[size_t(y+j)*rows+x] = tile[threadIdx.x][threadIdx.y+j];
}

__global__ void n8_pack_operands(uint4 const* src,uint4* dst,int rows,int cols) {
  __shared__ uint4 tile[32][33];
  int x=int(blockIdx.x)*32+int(threadIdx.x), y=int(blockIdx.y)*32+int(threadIdx.y);
  #pragma unroll
  for(int j=0;j<32;j+=8)
    tile[threadIdx.y+j][threadIdx.x]=x<cols && y+j<rows ? src[size_t(y+j)*cols+x] : make_uint4(0,0,0,0);
  __syncthreads();
  x=int(blockIdx.y)*32+int(threadIdx.x); y=int(blockIdx.x)*32+int(threadIdx.y);
  #pragma unroll
  for(int j=0;j<32;j+=8) if(x<rows && y+j<cols)
    dst[size_t(y+j)*rows+x]=tile[threadIdx.x][threadIdx.y+j];
}

template<int TN, int TK, bool Packed=false>
__global__ void n8k64_pipeline(uint8_t const* a, uint8_t const* b,
    uint32_t const* sa, uint32_t const* sb, uint8_t const* formats, BF16* d,
    int m, int n, int k, int fmt_a) {
  constexpr int NG=TN/8, KG=TK/64;
  constexpr int Need=TN+4+NG*4;
  constexpr int TC=Need<=32 ? 32 : Need<=64 ? 64 : Need<=128 ? 128 : Need<=256 ? 256 : 512;
  static_assert(Need<=512);
  __shared__ __align__(128) uint8_t as[2][128*TK/2];
  __shared__ __align__(128) uint8_t bs[2][TN*TK/2];
  __shared__ __align__(128) uint4 scales[2][KG][1+NG][32];
  __shared__ uint32_t tm;
  __shared__ __align__(16) uint64_t barrier;
  int tid=int(threadIdx.x), warp=tid/32;
  int row=int(blockIdx.x)*128, col=int(blockIdx.y)*TN;
  TMEM::Allocator1Sm allocator;
  if (warp==0) allocator.allocate(TC,&tm);
  if (tid==0) initialize_barrier(barrier,1);
  __syncthreads();
  if (warp==0) allocator.release_allocation_lock();
  auto descriptor=[](uint8_t* ptr,int rows) {
    UMMA::SmemDescriptor desc{};
    desc.version_=1;
    desc.start_address_=uint32_t(__cvta_generic_to_shared(ptr))>>4;
    desc.leading_byte_offset_=rows;
    desc.stride_byte_offset_=8;
    return desc;
  };
  auto load=[&](int kk,int stage) {
    if constexpr(Packed) {
      #pragma unroll
      for(int i=tid;i<128*TK/32;i+=128) {
        int r=i%128,chunk=i/128;
        bool valid=row+r<m && kk+chunk*32<k;
        auto src=a+(valid ? (size_t(kk/32+chunk)*m+row+r)*16 : 0);
        uint32_t dst=uint32_t(__cvta_generic_to_shared(as[stage]+i*16));
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(dst),"l"(src),"r"(valid?16:0));
      }
      #pragma unroll
      for(int i=tid;i<TN*TK/32;i+=128) {
        int c=i%TN,chunk=i/TN;
        bool valid=col+c<n && kk+chunk*32<k;
        auto src=b+(valid ? (size_t(kk/32+chunk)*n+col+c)*16 : 0);
        uint32_t dst=uint32_t(__cvta_generic_to_shared(bs[stage]+i*16));
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(dst),"l"(src),"r"(valid?16:0));
      }
      asm volatile("cp.async.commit_group;" ::: "memory");
    } else {
    // Adjacent lanes load adjacent K chunks, rather than distant rows.
    #pragma unroll
    for (int i=tid;i<128*TK/32;i+=128) {
      int r=i/(TK/32), chunk=i%(TK/32);
      uint4 v{};
      if (row+r<m && kk+chunk*32<k)
        v=*reinterpret_cast<uint4 const*>(a+size_t(row+r)*(k/2)+kk/2+chunk*16);
      reinterpret_cast<uint4*>(as[stage])[chunk*128+r]=v;
    }
    #pragma unroll
    for (int i=tid;i<TN*TK/32;i+=128) {
      int c=i/(TK/32), chunk=i%(TK/32);
      uint4 v{};
      if (col+c<n && kk+chunk*32<k)
        v=*reinterpret_cast<uint4 const*>(b+size_t(col+c)*(k/2)+kk/2+chunk*16);
      reinterpret_cast<uint4*>(bs[stage])[chunk*TN+c]=v;
    }
    }
    #pragma unroll
    for(int kg=0;kg<KG;++kg) {
      if(tid<32) {
        uint32_t v[4];
        #pragma unroll
        for(int q=0;q<4;++q) {
          int r=row+q*32+tid;
          v[q]=r<m && kk+kg*64<k ? sa[size_t(kk/64+kg)*m+r] : 0;
        }
        scales[stage][kg][0][tid]=make_uint4(v[0],v[1],v[2],v[3]);
      }
      #pragma unroll
      for(int i=tid;i<NG*32;i+=128) {
        int g=i/32,l=i%32,c=col+g*8+l;
        uint32_t v=l<8 && c<n && kk+kg*64<k ? sb[size_t(kk/64+kg)*n+c] : 0;
        scales[stage][kg][1+g][l]=make_uint4(v,0,0,0);
      }
    }
    cutlass::arch::fence_view_async_shared();
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
  };
  load(0,0);
  if constexpr(Packed) asm volatile("cp.async.wait_group 0;" ::: "memory");
  __syncthreads();
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
  auto idesc=UMMA::make_instr_desc_block_scaled<F4,F4,float,SF,128,8,
      UMMA::Major::K,UMMA::Major::K>();
  idesc.a_format_=uint8_t(fmt_a);
  int phase=0;
  for(int kk=0,stage=0;kk<k;kk+=TK,stage^=1) {
    if(warp==0) {
      #pragma unroll
      for(int kg=0;kg<KG;++kg) {
        if(kk+kg*64>=k) break;
        auto da=descriptor(as[stage]+kg*128*32,128);
        // tcgen05 copies and MMA share the ordered async thread. Scale buffers
        // can therefore be reused without reserving one set per pipeline stage.
        if(elect_one_sync()) {
          auto sd=descriptor(reinterpret_cast<uint8_t*>(scales[stage][kg][0]),32);
          SM100_UTCCP_4x32dp128bit_1cta::copy(sd,tm+TN);
          #pragma unroll
          for(int g=0;g<NG;++g) {
            auto bd=descriptor(reinterpret_cast<uint8_t*>(scales[stage][kg][1+g]),32);
            SM100_UTCCP_4x32dp128bit_1cta::copy(bd,tm+TN+4+g*4);
          }
        }
        #pragma unroll
        for(int g=0;g<NG;++g) {
          auto db=descriptor(bs[stage]+kg*TN*32+g*128,TN);
          int c=col+g*8;
          idesc.b_format_=c<n ? formats[size_t(c/8)*(k/64)+kk/64+kg] : 1;
          uint32_t sfa=tm+TN;
          uint32_t sfb=tm+TN+4+g*4;
          auto desc=UMMA::make_runtime_instr_desc_block_scaled(idesc,sfa,sfb);
          SM100_MMA_MXF4_SS<F4,F4,float,SF,128,8,16,
              UMMA::Major::K,UMMA::Major::K>::fma(da,db,tm+g*8,kk+kg*64!=0,desc,sfa,sfb);
        }
      }
      cutlass::arch::umma_arrive(&barrier);
    }
    // The next operands and scale factors use disjoint SMEM buffers.
    // Global loads overlap the already-issued MMA operations on this stage.
    if(kk+TK<k) load(kk+TK,stage^1);
    wait_barrier(barrier,phase);
    phase^=1;
    if constexpr(Packed) asm volatile("cp.async.wait_group 0;" ::: "memory");
    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
  }
  #pragma unroll
  for(int c=0;c<TN;c+=8) {
    uint32_t v[8];
    SM100_TMEM_LOAD_32dp32b8x::copy(tm+c+(warp*32u<<16),v[0],v[1],v[2],v[3],v[4],v[5],v[6],v[7]);
    cutlass::arch::fence_view_async_tmem_load();
    #pragma unroll
    for(int j=0;j<8;++j) if(row+tid<m && col+c+j<n)
      d[size_t(row+tid)*n+col+c+j]=BF16(__uint_as_float(v[j]));
  }
  __syncthreads();
  if(warp==0) allocator.free(tm,TC);
}
