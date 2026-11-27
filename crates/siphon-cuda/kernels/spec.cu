// Speculative decoding: the verify pass's feature taps, its greedy acceptance, and the
// rollback of the state that rejected rows changed.
//
// A verify runs m <= 8 rows (the anchor and m - 1 drafts) at positions start..start+m-1
// through chunked prefill. Each row appends to every KV cache, while conv states stay at
// their pre-verify values during the pass. After the acceptance keeps rows 0..a, two kinds
// of state need updating:
// - the sliding rings, where row j overwrote slot (start + j) mod slots, which held the
//   key of position start + j - slots. ring_backup saves those slots before the pass
//   and ring_restore puts back the rejected rows' slots after it. Global caches need
//   nothing: attention reads only slots below each row's position, and later rows
//   overwrite the stale ones.
// - the conv states, F32 [3, dims] per visit and stream (keys, values, post-attention,
//   post-MLP). During verify, each visit's conv kernels only read the pre-verify states
//   and its last kernel (moe_conv_residual_step_rows_2304 /
//   moe_conv_residual_batch_verify_2304 in conv.cu) records its m input rows into `inputs`
//   without modifying any conv state. After acceptance, conv_rollback shifts each state in
//   place to the last 3 rows of [old state; inputs of rows 0..a] without a backup buffer:
//   since keep = a + 1 >= 1, keep + r > r for all r in {0, 1, 2}, and for a >= 2 no old
//   state row is even read.
//
// Every kernel below (except opt-in accept_margin / accept_margin_batch) uses
// programmatic dependent launch (pdl.cuh), issuing pdl_trigger() and pdl_wait() before
// any global load and avoiding __restrict__ on pointers written by earlier kernels.

#include <cuda_bf16.h>
#include <cstdint>

#include "pdl.cuh"

namespace {

constexpr int kHidden = 2304;
constexpr int kKvDims = 1024;
constexpr int kVerifyRows = 8;
constexpr int kStateRows = 3;
constexpr int kStreams = 4;
// One row of conv inputs of a visit: keys, values, post-attention and post-MLP inputs.
constexpr int kVisitChannels = 2 * kKvDims + 2 * kHidden;

// Vectorized in-place conv rollback: 8 channels per thread (one 16-byte uint4 BF16 load
// from `inputs`, two 16-byte float4 loads/stores on F32 `state`).
constexpr int kRollbackVec = 8;
constexpr int kRollbackThreads = 96;
static_assert(kKvDims % kRollbackVec == 0, "kKvDims is 16-byte aligned for uint4/float4");
static_assert(kHidden % (kRollbackThreads * kRollbackVec) == 0, "blocks cover kHidden");
static_assert(kVisitChannels % kRollbackVec == 0, "kVisitChannels is 16-byte aligned");

__device__ __forceinline__ int stream_dims(int s) {
    return s < 2 ? kKvDims : kHidden;
}

// Offset of stream s in a row of kVisitChannels.
__device__ __forceinline__ int stream_offset(int s) {
    return s < 2 ? s * kKvDims : 2 * kKvDims + (s - 2) * kHidden;
}

__device__ __forceinline__ void bf16x8_to_f32x8(const __nv_bfloat16* p, float4& lo, float4& hi) {
    const uint4 packed = *reinterpret_cast<const uint4*>(p);
    const auto* vals = reinterpret_cast<const __nv_bfloat16*>(&packed);
    lo = make_float4(__bfloat162float(vals[0]), __bfloat162float(vals[1]),
                     __bfloat162float(vals[2]), __bfloat162float(vals[3]));
    hi = make_float4(__bfloat162float(vals[4]), __bfloat162float(vals[5]),
                     __bfloat162float(vals[6]), __bfloat162float(vals[7]));
}

__device__ __forceinline__ void rollback_channels_inplace(float* state, const __nv_bfloat16* rows,
                                                          int dims, int keep, int c) {
    float4 lo[kStateRows];
    float4 hi[kStateRows];
#pragma unroll
    for (int r = 0; r < kStateRows; ++r) {
        const int i = keep + r;
        if (i < kStateRows) {
            const float* src = state + size_t(i) * dims + c;
            lo[r] = *reinterpret_cast<const float4*>(src);
            hi[r] = *reinterpret_cast<const float4*>(src + 4);
        } else {
            bf16x8_to_f32x8(rows + size_t(i - kStateRows) * kVisitChannels + c, lo[r], hi[r]);
        }
    }
#pragma unroll
    for (int r = 0; r < kStateRows; ++r) {
        float* dst = state + size_t(r) * dims + c;
        *reinterpret_cast<float4*>(dst) = lo[r];
        *reinterpret_cast<float4*>(dst + 4) = hi[r];
    }
}

}  // namespace

// Copies rows of src [rows, 2304] into tap `tap` of taps [rows, count, 2304]. One block of
// 288 threads per row, 8 values per thread.
extern "C" __global__ void __launch_bounds__(288)
copy_tap(const __nv_bfloat16* src, __nv_bfloat16* taps, int tap, int count) {
    pdl_trigger();
    pdl_wait();
    const size_t r = blockIdx.x;
    const int c = threadIdx.x * 8;
    *reinterpret_cast<uint4*>(taps + (r * count + tap) * kHidden + c) =
        *reinterpret_cast<const uint4*>(src + r * kHidden + c);
}

// Shifts every conv state in place to rows keep..keep+2 of [state (3 rows); inputs (m rows)],
// where keep = result[0] + 1 rows of the verify were accepted. inputs is [visits, 8,
// kVisitChannels]. Grid: (3, 4, visits), 96 threads.
extern "C" __global__ void __launch_bounds__(kRollbackThreads)
conv_rollback(const unsigned long long* states, const __nv_bfloat16* inputs, const int* result) {
    pdl_trigger();
    pdl_wait();
    const int s = blockIdx.y;
    const int dims = stream_dims(s);
    const int c0 = (blockIdx.x * blockDim.x + threadIdx.x) * kRollbackVec;
    if (c0 >= dims) {
        return;
    }
    const int v = blockIdx.z;
    const int offset = stream_offset(s);
    const int keep = result[0] + 1;
    float* state = reinterpret_cast<float*>(states[v * kStreams + s]);
    const __nv_bfloat16* rows = inputs + size_t(v) * kVerifyRows * kVisitChannels + offset;
    const int stride = gridDim.x * blockDim.x * kRollbackVec;
    for (int c = c0; c < dims; c += stride) {
        rollback_channels_inplace(state, rows, dims, keep, c);
    }
}

// Saves the slots that rows 0..gridDim.x-1 of a verify at *start will overwrite in each
// sliding ring: slot (*start + j) % slots of each ring's keys and values (`row_bytes`
// bytes per slot: 2048 for BF16, 1024 for FP8, 576 for NVFP4). rings holds a keys and a
// values pointer per visit; backup is [visits, 8, 2, row_bytes]. Grid: (rows, visits),
// row_bytes / 16 threads.
extern "C" __global__ void __launch_bounds__(128)
ring_backup(const unsigned long long* rings, unsigned slots, const unsigned* start, uint8_t* backup,
            unsigned row_bytes) {
    pdl_trigger();
    pdl_wait();
    const int j = blockIdx.x;
    const int v = blockIdx.y;
    const unsigned byte_off = threadIdx.x * 16u;
    if (byte_off >= row_bytes) {
        return;
    }
    const size_t slot = (*start + j) % slots;
    const auto* keys = reinterpret_cast<const uint8_t*>(rings[2 * v]);
    const auto* values = reinterpret_cast<const uint8_t*>(rings[2 * v + 1]);
    uint8_t* dst = backup + (size_t(v) * kVerifyRows + j) * 2 * size_t(row_bytes);
    *reinterpret_cast<uint4*>(dst + byte_off) =
        *reinterpret_cast<const uint4*>(keys + slot * size_t(row_bytes) + byte_off);
    *reinterpret_cast<uint4*>(dst + row_bytes + byte_off) =
        *reinterpret_cast<const uint4*>(values + slot * size_t(row_bytes) + byte_off);
}

// Puts back the slots of the rejected rows j > result[0], from ring_backup's copy.
extern "C" __global__ void __launch_bounds__(128)
ring_restore(const unsigned long long* rings, unsigned slots, const unsigned* start, const uint8_t* backup,
             const int* result, unsigned row_bytes) {
    pdl_trigger();
    pdl_wait();
    const int j = blockIdx.x;
    if (j <= result[0]) {
        return;
    }
    const unsigned byte_off = threadIdx.x * 16u;
    if (byte_off >= row_bytes) {
        return;
    }
    const int v = blockIdx.y;
    const size_t slot = (*start + j) % slots;
    auto* keys = reinterpret_cast<uint8_t*>(rings[2 * v]);
    auto* values = reinterpret_cast<uint8_t*>(rings[2 * v + 1]);
    const uint8_t* src = backup + (size_t(v) * kVerifyRows + j) * 2 * size_t(row_bytes);
    *reinterpret_cast<uint4*>(keys + slot * size_t(row_bytes) + byte_off) =
        *reinterpret_cast<const uint4*>(src + byte_off);
    *reinterpret_cast<uint4*>(values + slot * size_t(row_bytes) + byte_off) =
        *reinterpret_cast<const uint4*>(src + row_bytes + byte_off);
}

// Greedy acceptance of a verify of m rows: tokens [m] holds the anchor and m - 1 drafts,
// argmax [m] each row's greedy token. Accepts the longest prefix of drafts where draft i
// equals argmax[i - 1], or exactly *keep - 1 drafts when *keep is nonzero. Writes
// result = [accepted, tokens[0..8], argmax[0..8]] (rows past m left alone) and the next
// anchor, argmax[accepted], to anchor. One thread.
extern "C" __global__ void accept_greedy(const int* tokens, const int* argmax, const unsigned* keep, int m,
                                         int* result, int* anchor) {
    pdl_trigger();
    pdl_wait();
    int accepted = 0;
    if (*keep != 0) {
        accepted = min(int(*keep), m) - 1;
    } else {
        while (accepted + 1 < m && tokens[accepted + 1] == argmax[accepted]) {
            ++accepted;
        }
    }
    result[0] = accepted;
    for (int i = 0; i < m; ++i) {
        result[1 + i] = tokens[i];
        result[1 + kVerifyRows + i] = argmax[i];
    }
    *anchor = argmax[accepted];
}

// Margin-tolerant speculative acceptance of a verify of m rows: tokens [m] holds the anchor
// and m - 1 drafts, argmax [m] each row's target token (greedy argmax or sampled draw), and
// logits [m, vocab] each row's F32 softcapped target logits. Accepts the longest prefix of
// drafts where draft == target, or (margin > 0 and logits[accepted, draft] >=
// logits[accepted, target] - margin), or exactly *keep - 1 drafts when *keep is nonzero.
// Writes result = [accepted, tokens[0..8], argmax[0..8]] (rows past m left alone) and the
// next anchor, argmax[accepted], to anchor. One thread.
extern "C" __global__ void accept_margin(const int* tokens, const int* argmax, const float* logits,
                                         int vocab, const unsigned* keep, int m, float margin,
                                         int* result, int* anchor) {
    int accepted = 0;
    if (*keep != 0) {
        accepted = min(int(*keep), m) - 1;
    } else {
        while (accepted + 1 < m) {
            const int draft = tokens[accepted + 1];
            const int target = argmax[accepted];
            if (draft == target) {
                ++accepted;
                continue;
            }
            if (margin > 0.f && draft >= 0 && draft < vocab && target >= 0 && target < vocab) {
                const float* row = logits + size_t(accepted) * size_t(vocab);
                if (row[draft] >= row[target] - margin) {
                    ++accepted;
                    continue;
                }
            }
            break;
        }
    }
    result[0] = accepted;
    for (int i = 0; i < m; ++i) {
        result[1 + i] = tokens[i];
        result[1 + kVerifyRows + i] = argmax[i];
    }
    *anchor = argmax[accepted];
}

// ---------------------------------------------------------------------------------------
// Batched verify: `seqs` sequences verified in parallel, each with `m` (1..=8) rows, using
// the visit state table `tables` [visits, max_seqs, 6] (words 0..1 KV caches, 2..5 conv
// states).

namespace {
constexpr int kBatchStateWords = 6;
constexpr int kResultLen = 1 + 2 * kVerifyRows;
}  // namespace

// Shifts every (visit, seq) conv state in place from its pre-verify values and inputs
// [visits, max_seqs, 8, kVisitChannels] using results [seqs, kResultLen].
// Grid: (3, 4, visits * seqs), 96 threads.
extern "C" __global__ void __launch_bounds__(kRollbackThreads)
conv_rollback_batch(const unsigned long long* tables, int max_seqs, int seqs,
                    const __nv_bfloat16* inputs, const int* results) {
    pdl_trigger();
    pdl_wait();
    const int s = blockIdx.y;
    const int dims = stream_dims(s);
    const int c0 = (blockIdx.x * blockDim.x + threadIdx.x) * kRollbackVec;
    if (c0 >= dims) {
        return;
    }
    const int v = blockIdx.z / seqs;
    const int seq = blockIdx.z % seqs;
    const int offset = stream_offset(s);
    const int keep = results[size_t(seq) * kResultLen] + 1;
    const size_t entry = size_t(v) * size_t(max_seqs) + size_t(seq);
    float* state = reinterpret_cast<float*>(tables[entry * kBatchStateWords + 2 + s]);
    const __nv_bfloat16* rows = inputs + entry * kVerifyRows * kVisitChannels + offset;
    const int stride = gridDim.x * blockDim.x * kRollbackVec;
    for (int c = c0; c < dims; c += stride) {
        rollback_channels_inplace(state, rows, dims, keep, c);
    }
}

// Saves the sliding ring slots that rows 0..gridDim.x-1 of each sequence overwrite:
// sliding_visits [num_sliding] maps sliding index sv -> visit index v in `tables`.
// backup is [num_sliding, max_seqs, 8, 2, row_bytes] bytes. Grid: (rows, num_sliding, seqs),
// row_bytes / 16 threads.
extern "C" __global__ void __launch_bounds__(128)
ring_backup_batch(const unsigned long long* tables, const int* sliding_visits, int max_seqs, unsigned slots,
                  const unsigned* positions, uint8_t* backup, unsigned row_bytes) {
    pdl_trigger();
    pdl_wait();
    const int j = blockIdx.x;
    const int sv = blockIdx.y;
    const int seq = blockIdx.z;
    const unsigned byte_off = threadIdx.x * 16u;
    if (byte_off >= row_bytes) {
        return;
    }
    const int v = sliding_visits[sv];
    const size_t slot = (positions[seq] + j) % slots;
    const size_t entry = size_t(v) * size_t(max_seqs) + size_t(seq);
    const auto* keys = reinterpret_cast<const uint8_t*>(tables[entry * kBatchStateWords]);
    const auto* values = reinterpret_cast<const uint8_t*>(tables[entry * kBatchStateWords + 1]);
    uint8_t* dst =
        backup + ((size_t(sv) * size_t(max_seqs) + size_t(seq)) * kVerifyRows + j) * 2 * size_t(row_bytes);
    *reinterpret_cast<uint4*>(dst + byte_off) =
        *reinterpret_cast<const uint4*>(keys + slot * size_t(row_bytes) + byte_off);
    *reinterpret_cast<uint4*>(dst + row_bytes + byte_off) =
        *reinterpret_cast<const uint4*>(values + slot * size_t(row_bytes) + byte_off);
}

// Restores rejected sliding ring slots j > results[seq * kResultLen] for every sequence.
// Grid: (rows, num_sliding, seqs), row_bytes / 16 threads.
extern "C" __global__ void __launch_bounds__(128)
ring_restore_batch(const unsigned long long* tables, const int* sliding_visits, int max_seqs, unsigned slots,
                   const unsigned* positions, const uint8_t* backup, const int* results, unsigned row_bytes) {
    pdl_trigger();
    pdl_wait();
    const int j = blockIdx.x;
    const int seq = blockIdx.z;
    if (j <= results[size_t(seq) * kResultLen]) {
        return;
    }
    const unsigned byte_off = threadIdx.x * 16u;
    if (byte_off >= row_bytes) {
        return;
    }
    const int sv = blockIdx.y;
    const int v = sliding_visits[sv];
    const size_t slot = (positions[seq] + j) % slots;
    const size_t entry = size_t(v) * size_t(max_seqs) + size_t(seq);
    auto* keys = reinterpret_cast<uint8_t*>(tables[entry * kBatchStateWords]);
    auto* values = reinterpret_cast<uint8_t*>(tables[entry * kBatchStateWords + 1]);
    const uint8_t* src =
        backup + ((size_t(sv) * size_t(max_seqs) + size_t(seq)) * kVerifyRows + j) * 2 * size_t(row_bytes);
    *reinterpret_cast<uint4*>(keys + slot * size_t(row_bytes) + byte_off) =
        *reinterpret_cast<const uint4*>(src + byte_off);
    *reinterpret_cast<uint4*>(values + slot * size_t(row_bytes) + byte_off) =
        *reinterpret_cast<const uint4*>(src + row_bytes + byte_off);
}

// Greedy acceptance for `seqs` sequences, each of `m` rows. tokens and argmax are [seqs, m];
// keep is [seqs] (0 for greedy, else exact rows to keep); results is [seqs, kResultLen];
// anchors is [seqs]. Grid: (seqs, 1, 1), 1 thread.
extern "C" __global__ void accept_greedy_batch(const int* tokens, const int* argmax, const unsigned* keep,
                                               int m, int* results, int* anchors) {
    pdl_trigger();
    pdl_wait();
    const int seq = blockIdx.x;
    const int* seq_tokens = tokens + size_t(seq) * size_t(m);
    const int* seq_argmax = argmax + size_t(seq) * size_t(m);
    int* result = results + size_t(seq) * kResultLen;
    int accepted = 0;
    const unsigned k = keep[seq];
    if (k != 0) {
        accepted = min(int(k), m) - 1;
    } else {
        while (accepted + 1 < m && seq_tokens[accepted + 1] == seq_argmax[accepted]) {
            ++accepted;
        }
    }
    result[0] = accepted;
    for (int i = 0; i < m; ++i) {
        result[1 + i] = seq_tokens[i];
        result[1 + kVerifyRows + i] = seq_argmax[i];
    }
    anchors[seq] = seq_argmax[accepted];
}

// Margin-tolerant acceptance for `seqs` sequences, each of `m` rows. tokens and argmax are
// [seqs, m]; logits is [seqs, m, vocab] (F32 softcapped target logits); keep is [seqs]
// (0 for margin acceptance, else exact rows to keep); results is [seqs, kResultLen];
// anchors is [seqs]. Grid: (seqs, 1, 1), 1 thread.
extern "C" __global__ void accept_margin_batch(const int* tokens, const int* argmax,
                                               const float* logits, int vocab,
                                               const unsigned* keep, int m, float margin,
                                               int* results, int* anchors) {
    const int seq = blockIdx.x;
    const int* seq_tokens = tokens + size_t(seq) * size_t(m);
    const int* seq_argmax = argmax + size_t(seq) * size_t(m);
    const float* seq_logits = logits + size_t(seq) * size_t(m) * size_t(vocab);
    int* result = results + size_t(seq) * kResultLen;
    int accepted = 0;
    const unsigned k = keep[seq];
    if (k != 0) {
        accepted = min(int(k), m) - 1;
    } else {
        while (accepted + 1 < m) {
            const int draft = seq_tokens[accepted + 1];
            const int target = seq_argmax[accepted];
            if (draft == target) {
                ++accepted;
                continue;
            }
            if (margin > 0.f && draft >= 0 && draft < vocab && target >= 0 && target < vocab) {
                const float* row = seq_logits + size_t(accepted) * size_t(vocab);
                if (row[draft] >= row[target] - margin) {
                    ++accepted;
                    continue;
                }
            }
            break;
        }
    }
    result[0] = accepted;
    for (int i = 0; i < m; ++i) {
        result[1 + i] = seq_tokens[i];
        result[1 + kVerifyRows + i] = seq_argmax[i];
    }
    anchors[seq] = seq_argmax[accepted];
}

// Packs [seqs, kVerifyRows] drafted verify tokens into contiguous [seqs, width] rows for
// batched verify. One block of 256 threads (seqs * width <= 256). Launched with PDL.
extern "C" __global__ void __launch_bounds__(256)
pack_verify_tokens_batch(const int* src, int* dst, int seqs, int width) {
    pdl_wait();
    pdl_trigger();
    const int total = seqs * width;
    for (int idx = threadIdx.x; idx < total; idx += blockDim.x) {
        const int seq = idx / width;
        const int col = idx % width;
        dst[idx] = src[size_t(seq) * kVerifyRows + col];
    }
}

// ---------------------------------------------------------------------------------------
// 2-branch speculative tree verify: acceptance along primary/secondary child pointers,
// winning-path KV commit (sliding and global, all formats), conv state commit, and
// feature tap compaction for the draft model.

// Tree acceptance over m <= 8 nodes in topological order:
// - If *forced_len != 0, copies forced_path[0..*forced_len] into path and accepts
//   *forced_len - 1 draft tokens (teacher forcing / parity testing).
// - Otherwise walks from root 0: checks primary_child[u] first, then secondary_child[u].
// Writes result = [accepted, tokens[0..m], argmax[0..m]], path[0..=accepted], and
// *anchor = argmax[path[accepted]]. One thread.
extern "C" __global__ void accept_tree(const int* tokens, const int* argmax,
                                       const int* primary_child, const int* secondary_child,
                                       const int* forced_path, const unsigned* forced_len, int m,
                                       int* result, int* path, int* anchor) {
    int accepted = 0;
    path[0] = 0;
    const unsigned flen = *forced_len;
    if (flen != 0) {
        const int keep = min(int(flen), m);
        accepted = keep - 1;
        for (int d = 0; d < keep; ++d) {
            path[d] = forced_path[d];
        }
    } else {
        int u = 0;
        while (accepted + 1 < m) {
            const int target = argmax[u];
            const int c1 = primary_child[u];
            const int c2 = secondary_child[u];
            if (c1 >= 0 && tokens[c1] == target) {
                u = c1;
                path[++accepted] = u;
            } else if (c2 >= 0 && tokens[c2] == target) {
                u = c2;
                path[++accepted] = u;
            } else {
                break;
            }
        }
    }
    result[0] = accepted;
    for (int i = 0; i < m; ++i) {
        result[1 + i] = tokens[i];
        result[1 + kVerifyRows + i] = argmax[i];
    }
    *anchor = argmax[path[accepted]];
}

// Commits the winning tree path's staged K and V rows into every visit's KV cache
// (both sliding rings and global caches, BF16 / FP8 / NVFP4).
// caches[2*v], caches[2*v+1]: device pointers to visit v's k_cache and v_cache.
// stages[2*v], stages[2*v+1]: device pointers to visit v's [8, row_bytes] staged K and V.
// slots[v]: cache slot count of visit v; row_bytes[v]: bytes per slot (2048, 1024, or 576).
// Grid: (max_depth_plus_1, visits, 1), 128 threads (16 bytes per thread, covers 2048 B).
extern "C" __global__ void __launch_bounds__(128)
kv_commit_tree(const unsigned long long* caches, const unsigned long long* stages,
               const unsigned* slots, const unsigned* row_bytes, const unsigned* start,
               const int* result, const int* path) {
    const int d = blockIdx.x;
    if (d > result[0]) {
        return;
    }
    const int v = blockIdx.y;
    const unsigned rb = row_bytes[v];
    const unsigned byte_off = threadIdx.x * 16u;
    if (byte_off >= rb) {
        return;
    }
    const int src_row = path[d];
    const size_t dst_slot = (*start + unsigned(d)) % slots[v];
    const auto* src_k = reinterpret_cast<const uint8_t*>(stages[2 * v]);
    const auto* src_v = reinterpret_cast<const uint8_t*>(stages[2 * v + 1]);
    auto* dst_k = reinterpret_cast<uint8_t*>(caches[2 * v]);
    auto* dst_v = reinterpret_cast<uint8_t*>(caches[2 * v + 1]);
    *reinterpret_cast<uint4*>(dst_k + dst_slot * size_t(rb) + byte_off) =
        *reinterpret_cast<const uint4*>(src_k + size_t(src_row) * size_t(rb) + byte_off);
    *reinterpret_cast<uint4*>(dst_v + dst_slot * size_t(rb) + byte_off) =
        *reinterpret_cast<const uint4*>(src_v + size_t(src_row) * size_t(rb) + byte_off);
}

// Shifts every visit's four conv states in place from [state (3 rows); inputs[path[0..=accepted]]].
// Tree verify leaves `state` untouched during the forward pass, so no separate backup is needed.
// Grid: (3, 4, visits), 96 threads (vectorized 8 channels/thread matching `conv_rollback`).
extern "C" __global__ void __launch_bounds__(kRollbackThreads)
conv_commit_tree(const unsigned long long* states, const __nv_bfloat16* inputs,
                 const int* result, const int* path) {
    pdl_trigger();
    pdl_wait();
    const int s = blockIdx.y;
    const int dims = stream_dims(s);
    const int c0 = (blockIdx.x * blockDim.x + threadIdx.x) * kRollbackVec;
    if (c0 >= dims) {
        return;
    }
    const int v = blockIdx.z;
    const int offset = stream_offset(s);
    const int keep = result[0] + 1;
    float* state = reinterpret_cast<float*>(states[v * kStreams + s]);
    const __nv_bfloat16* rows = inputs + size_t(v) * kVerifyRows * kVisitChannels + offset;
    const int stride = gridDim.x * blockDim.x * kRollbackVec;
    for (int c = c0; c < dims; c += stride) {
        float4 lo[kStateRows];
        float4 hi[kStateRows];
#pragma unroll
        for (int r = 0; r < kStateRows; ++r) {
            const int i = keep + r;
            if (i < kStateRows) {
                const float* src = state + size_t(i) * dims + c;
                lo[r] = *reinterpret_cast<const float4*>(src);
                hi[r] = *reinterpret_cast<const float4*>(src + 4);
            } else {
                const int src_row = path[i - kStateRows];
                bf16x8_to_f32x8(rows + size_t(src_row) * kVisitChannels + c, lo[r], hi[r]);
            }
        }
#pragma unroll
        for (int r = 0; r < kStateRows; ++r) {
            float* dst = state + size_t(r) * dims + c;
            *reinterpret_cast<float4*>(dst) = lo[r];
            *reinterpret_cast<float4*>(dst + 4) = hi[r];
        }
    }
}

// Compacts feature taps along the winning path path[0..=accepted] into contiguous rows
// 0..=accepted of dst_taps, so the draft model's `set_round_inputs_kept` commits them
// unchanged. Grid: (max_depth_plus_1, tap_count, 1), 288 threads.
extern "C" __global__ void __launch_bounds__(288)
taps_compact_tree(const __nv_bfloat16* src_taps, __nv_bfloat16* dst_taps, const int* result,
                  const int* path, int tap_count) {
    const int d = blockIdx.x;
    if (d > result[0]) {
        return;
    }
    const int tap = blockIdx.y;
    const int src_row = path[d];
    const int c = threadIdx.x * 8;
    *reinterpret_cast<uint4*>(dst_taps + (size_t(d) * tap_count + tap) * kHidden + c) =
        *reinterpret_cast<const uint4*>(src_taps + (size_t(src_row) * tap_count + tap) * kHidden + c);
}
