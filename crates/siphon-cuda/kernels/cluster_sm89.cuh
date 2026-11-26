// Thread-block clusters for Ada (sm_89), which has none: the cluster kernels' exchanges
// through distributed shared memory go through global memory instead, with the same
// values summed in the same order, so the results equal the sm_120a build's bit for bit.
//
// A cluster is `size` consecutive blocks along x. The blocks of a cluster are ordinary
// blocks here, and they wait for each other, so they must all be resident at once: every
// grid that uses this keeps its clusters at 16 blocks or fewer, and the 4090's 128 SMs
// start blocks in index order, so the earliest unfinished cluster always has every block
// running.
//
// Two exchanges, matching the sm_120a kernels':
// - Mail stands in for st.async stores into another block's shared memory that complete
//   on its mbarrier. Each block has a mailbox in global memory that mirrors its shared
//   memory by address: a send writes the values at the target's address in the receiver's
//   mailbox, then adds their bytes to the 32-bit word at the address of the receiver's
//   mbarrier. The receiver waits for the bytes it expects, copies the values into its own
//   shared memory, and sets the word back to 0 for the next launch. Every barrier that
//   receives a send must be waited on in the same launch.
// - barrier() stands in for barrier.cluster arrive and wait: a sense-reversing barrier on
//   two counters per cluster, as cooperative groups' grid sync works.

#pragma once

#include "arch.cuh"

#if SIPHON_SM89

#include <cstdint>

namespace sm89 {

// The cluster's index in the grid.
__device__ __forceinline__ int cluster_index(int size) {
    return int(blockIdx.x) / size + int(gridDim.x) / size * int(blockIdx.y + gridDim.y * blockIdx.z);
}

__device__ __forceinline__ uint32_t smem_address(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ unsigned load_acquire(const unsigned* p) {
    unsigned v;
    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}

// Per-block mailboxes of kBytes, for up to kBlocks blocks.
template <int kBlocks, int kBytes>
struct Mailboxes {
    static_assert(kBytes % 16 == 0, "16-byte rows");
    uint4 box[kBlocks][kBytes / 16];

    // The address in block `block`'s mailbox that mirrors shared address `addr`, for
    // `bytes` bytes. Traps past the mailboxes, which the launch shapes rule out.
    __device__ __forceinline__ char* at(int block, uint32_t addr, uint32_t bytes) {
        if (block >= kBlocks || addr + bytes > uint32_t(kBytes)) {
            __trap();
        }
        return reinterpret_cast<char*>(box[block]) + addr;
    }

    // Stores `words` (2 or 4 words) at shared address `target` of block `rank` of this
    // block's cluster, completing on that block's barrier at shared address `barrier`.
    template <int kWords>
    __device__ __forceinline__ void send(int size, int rank, uint32_t target, uint32_t barrier,
                                         const uint32_t (&words)[kWords]) {
        static_assert(kWords == 2 || kWords == 4, "v2 or v4 stores");
        const int block = cluster_index(size) * size + rank;
        char* dst = at(block, target, 4 * kWords);
        if constexpr (kWords == 4) {
            *reinterpret_cast<uint4*>(dst) = make_uint4(words[0], words[1], words[2], words[3]);
        } else {
            *reinterpret_cast<uint2*>(dst) = make_uint2(words[0], words[1]);
        }
        __threadfence();
        atomicAdd(reinterpret_cast<unsigned*>(at(block, barrier, 4)), unsigned(4 * kWords));
    }

    // Waits on this block's barrier at shared address `barrier` until `bytes` have
    // arrived, then copies them from the mailbox to shared memory at `target`. Every
    // thread of the calling warp calls it, or of the block with kBlock.
    template <bool kBlock>
    __device__ __forceinline__ void wait(int size, uint32_t barrier, void* target, uint32_t bytes) {
        const int block = cluster_index(size) * size + int(blockIdx.x) % size;
        auto* count = reinterpret_cast<unsigned*>(at(block, barrier, 4));
        const int t = kBlock ? int(threadIdx.x) : int(threadIdx.x) % 32;
        const int threads = kBlock ? int(blockDim.x) : 32;
        if (t == 0) {
            while (load_acquire(count) < bytes) {
            }
        }
        if constexpr (kBlock) {
            __syncthreads();
        } else {
            __syncwarp();
        }
        const auto* src = reinterpret_cast<const uint32_t*>(at(block, smem_address(target), bytes));
        auto* dst = static_cast<uint32_t*>(target);
        for (uint32_t i = t; i < bytes / 4; i += threads) {
            dst[i] = __ldcg(src + i);
        }
        if constexpr (kBlock) {
            __syncthreads();
        } else {
            __syncwarp();
        }
        if (t == 0) {
            *count = 0;
        }
    }
};

// Counters of a sense-reversing barrier.
struct BarrierState {
    unsigned count;
    unsigned generation;
};

// Every thread of each of the `size` blocks of the cluster calls it. Writes to global
// memory before it are visible to the cluster's blocks after it.
__device__ __forceinline__ void barrier(BarrierState* states, int max_clusters, int size) {
    const int c = cluster_index(size);
    if (c >= max_clusters) {
        __trap();
    }
    BarrierState& s = states[c];
    __syncthreads();
    if (threadIdx.x == 0) {
        // The generation cannot advance before this block arrives.
        const unsigned generation = load_acquire(&s.generation);
        __threadfence();
        if (atomicAdd(&s.count, 1u) == unsigned(size) - 1) {
            atomicExch(&s.count, 0u);
            __threadfence();
            atomicAdd(&s.generation, 1u);
        } else {
            while (load_acquire(&s.generation) == generation) {
            }
        }
    }
    __syncthreads();
}

}  // namespace sm89

#endif
