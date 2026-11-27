// The noise of the exponential race that draws a sampled token (sample.cu), shared with
// the draft's coupled picks (draft.cu), so that the two cannot diverge.
//
// Token i of a row scores x_i / T - ln E_i, with E_i an Exp(1) variable hashed from the
// request's seed, the position the token takes and i. sample_race keeps the best score
// among the filtered tokens. A sampling draft takes the best score over its own logits
// with the same seed and position, so it proposes the target's draw whenever its logits
// are close enough to the target's (docs/engine.md, "Speculative sampling").
// crates/siphon/src/race.rs ports these functions to the host.

#pragma once

namespace race {

// More than the largest -ln E_i can reach: E_i >= 2^-65, so -ln E_i < 45.1.
constexpr float kMaxBoost = 46.0f;

__device__ __forceinline__ unsigned long long mix64(unsigned long long z) {
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}

// The key of a row's draw: its seed and the position its token takes.
__device__ __forceinline__ unsigned long long draw_key(unsigned seed_lo, unsigned seed_hi,
                                                       unsigned position) {
    const unsigned long long seed = (static_cast<unsigned long long>(seed_hi) << 32) | seed_lo;
    return mix64(seed ^ mix64(position + 0x632be59bd9b4e019ull));
}

// z / 2^64 for z below 2^63, at least 2^-65, from two 32-bit conversions: 64-bit ones run
// at the FP64 rate on this GPU.
__device__ __forceinline__ float unit(unsigned long long z) {
    const float v = static_cast<float>(static_cast<unsigned>(z >> 32)) * 0x1p-32f +
                    static_cast<float>(static_cast<unsigned>(z)) * 0x1p-64f;
    return fmaxf(v, 0x1p-65f);
}

// -ln E_i for an Exp(1) variable E_i. 64 bits split into a uniform v below 1/2 or its
// complement, so that both small v (small E) and v near 1 keep 24 significant bits.
__device__ __forceinline__ float race_boost(unsigned long long key, int i) {
    const unsigned long long z = mix64(key + (static_cast<unsigned long long>(i) + 1) * 0x9e3779b97f4a7c15ull);
    float e;
    if ((z >> 63) == 0) {
        e = -log1pf(-unit(z));
    } else {
        e = -logf(unit(~z));
    }
    return -logf(e);
}

// Independent RNG domain-separation salts for beta-relaxed / maximal-coupling rejection
// sampling (sample.cu): acceptance uniform draw vs residual resampling race.
constexpr unsigned long long kAcceptSalt = 0x517cc1b727220a95ull;
constexpr unsigned long long kResidSalt = 0x94d049bb133111ebull;

// A request's sampling as the draft reads it, RACE_WORDS (= 8) in ops/draft.rs:
// temperature 0 drafts greedily. DFlash 2's 16-candidate selector applies the target's
// filters (top_k, top_p, min_p) and draft temperature scale `temp_scale` (default 0.85)
// in registers before the coupled Gumbel-Max race, preserving exact target losslessness.
struct DraftRace {
    float temperature;
    unsigned seed_lo;
    unsigned seed_hi;
    int top_k;
    float top_p;
    float min_p;
    float temp_scale;
    unsigned pad;
};

}  // namespace race
