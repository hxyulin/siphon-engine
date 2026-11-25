//! The exponential race that draws a sampled token, ported from `kernels/race.cuh` in
//! siphon-cuda for host-side measurements and tests.
//!
//! Token `i` of a row scores `x_i / T - ln E_i`, with `E_i` an Exp(1) variable hashed from
//! the request's seed, the position the token takes and `i`; the best score among the
//! kept tokens is a draw from the filtered distribution. A coupled draft takes the best
//! score over its own logits with the same noise.
//!
//! [`mix64`], [`draw_key`] and [`unit`] are bit-exact ports. [`race_boost`] takes three
//! logarithms, which the GPU computes with its own `logf` and `log1pf` (within 1 ulp,
//! not correctly rounded), so it can differ from the kernel's value in the last bits.

use std::cmp::Ordering;

/// More than the largest `-ln E_i` can reach: `E_i >= 2^-65`, so `-ln E_i < 45.1`.
pub const MAX_BOOST: f32 = 46.0;

/// splitmix64's finalizer.
#[must_use]
pub fn mix64(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// The key of a draw: the request's seed and the position its token takes.
#[must_use]
pub fn draw_key(seed: u64, position: u32) -> u64 {
    mix64(seed ^ mix64(u64::from(position).wrapping_add(0x632b_e59b_d9b4_e019)))
}

/// `z / 2^64` for `z` below `2^63`, at least `2^-65`, from two 32-bit conversions as the
/// kernel does them.
#[must_use]
pub fn unit(z: u64) -> f32 {
    let hi = (z >> 32) as u32 as f32;
    let lo = z as u32 as f32;
    // Both products are exact, so the kernel's possible fused multiply-add gives the
    // same sum.
    (hi * f32::from_bits(0x2f80_0000) + lo * f32::from_bits(0x1f80_0000))
        .max(f32::from_bits(0x1f00_0000))
}

/// `-ln E_i` for token `i` of the draw with `key`.
#[must_use]
pub fn race_boost(key: u64, i: u32) -> f32 {
    let z = mix64(key.wrapping_add((u64::from(i) + 1).wrapping_mul(0x9e37_79b9_7f4a_7c15)));
    let e = if z >> 63 == 0 {
        -(-unit(z)).ln_1p()
    } else {
        -unit(!z).ln()
    };
    -e.ln()
}

/// The race's score of a token with logit `logit`: `logit * inv_t + race_boost`, rounded
/// after each operation as the draft's kernels compute it.
#[must_use]
pub fn score(logit: f32, inv_t: f32, key: u64, i: u32) -> f32 {
    logit * inv_t + race_boost(key, i)
}

/// The winner of the race over `candidates` (token ids into `logits`) at temperature
/// `temperature` with `key`: the best [`score`], ties to the smaller id, NaN logits never
/// winning. `None` without candidates.
#[must_use]
pub fn pick(
    logits: &[f32],
    candidates: impl IntoIterator<Item = usize>,
    temperature: f32,
    key: u64,
) -> Option<usize> {
    let inv_t = 1.0 / temperature;
    let mut best: Option<(f32, usize)> = None;
    for i in candidates {
        let logit = logits[i];
        if logit.is_nan() {
            continue;
        }
        let s = score(logit, inv_t, key, u32::try_from(i).ok()?);
        let better = match best {
            None => true,
            Some((b, j)) => match s.partial_cmp(&b) {
                Some(Ordering::Greater) => true,
                Some(Ordering::Equal) => i < j,
                _ => false,
            },
        };
        if better {
            best = Some((s, i));
        }
    }
    best.map(|(_, i)| i)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mix64_is_splitmix64() {
        // splitmix64 seeded with 0: the first outputs mix 0x9e37_79b9_7f4a_7c15 times 1, 2, 3.
        let mut state = 0u64;
        let mut next = || {
            state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
            mix64(state)
        };
        assert_eq!(next(), 0xe220_a839_7b1d_cdaf);
        assert_eq!(next(), 0x6e78_9e6a_a1b9_65f4);
        assert_eq!(next(), 0x06c4_5d18_8009_454f);
    }

    #[test]
    fn unit_covers_the_interval() {
        assert_eq!(unit(0), f32::from_bits(0x1f00_0000));
        assert_eq!(unit(1 << 62), 0.25);
        assert!(unit((1 << 63) - 1) <= 0.5);
        assert_eq!(unit(1 << 32), 2f32.powi(-32));
    }

    #[test]
    fn boosts_are_gumbel() {
        // -ln E for E ~ Exp(1) is a standard Gumbel variable: mean 0.5772, variance pi^2 / 6.
        let key = draw_key(42, 1000);
        let n = 200_000u32;
        let values: Vec<f64> = (0..n).map(|i| f64::from(race_boost(key, i))).collect();
        let mean = values.iter().sum::<f64>() / f64::from(n);
        let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / f64::from(n);
        assert!((mean - 0.577_215_66).abs() < 0.01, "mean {mean}");
        assert!((var - 1.644_934).abs() < 0.03, "variance {var}");
        assert!(values.iter().all(|&v| v < f64::from(MAX_BOOST)));
    }

    #[test]
    fn picks_follow_the_distribution() {
        // Three tokens with probabilities 0.5, 0.3, 0.2 at temperature 1.
        let logits = [0.5f32.ln(), 0.3f32.ln(), 0.2f32.ln()];
        let mut counts = [0u32; 3];
        for position in 0..100_000 {
            let i = pick(&logits, 0..3, 1.0, draw_key(7, position)).unwrap_or(9);
            counts[i] += 1;
        }
        for (count, p) in counts.iter().zip([0.5, 0.3, 0.2]) {
            let share = f64::from(*count) / 100_000.0;
            assert!((share - p).abs() < 0.01, "{counts:?}");
        }
        // Equal logits draw the same token as the noise alone ranks.
        assert_eq!(
            pick(&[1.0, 1.0], [1, 0], 1.0, 3),
            pick(&[0.0, 0.0], 0..2, 2.0, 3)
        );
    }
}
