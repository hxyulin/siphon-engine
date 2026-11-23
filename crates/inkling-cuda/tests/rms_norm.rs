//! Compares the RMSNorm kernel with a CPU reference that uses the same rounding steps.

#![cfg(target_os = "linux")]

use half::bf16;
use inkling_cuda::Gpu;

const HIDDEN: usize = 2304;
const EPS: f32 = 1e-6;

/// Deterministic values in roughly [-scale, scale], from a xorshift generator.
fn random_bf16(len: usize, seed: u64, scale: f32) -> Vec<bf16> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let unit = (state >> 40) as f32 / (1u64 << 24) as f32;
            bf16::from_f32((unit * 2.0 - 1.0) * scale)
        })
        .collect()
}

fn reference(x: &[bf16], weight: &[bf16]) -> Vec<bf16> {
    x.chunks(HIDDEN)
        .flat_map(|row| {
            let sum: f64 = row.iter().map(|v| f64::from(v.to_f32()).powi(2)).sum();
            let inv_rms = (1.0 / (sum / HIDDEN as f64 + f64::from(EPS)).sqrt()) as f32;
            row.iter().zip(weight).map(move |(v, w)| {
                bf16::from_f32(bf16::from_f32(v.to_f32() * inv_rms).to_f32() * w.to_f32())
            })
        })
        .collect()
}

#[test]
fn rms_norm_matches_reference() -> anyhow::Result<()> {
    let gpu = Gpu::new(0)?;
    let stream = gpu.stream();
    let rows = 37;
    let x = random_bf16(rows * HIDDEN, 1, 4.0);
    let weight = random_bf16(HIDDEN, 2, 1.5);

    let x_dev = stream.clone_htod(&x)?;
    let weight_dev = stream.clone_htod(&weight)?;
    let mut out_dev = stream.alloc_zeros::<bf16>(x.len())?;
    gpu.rms_norm(&x_dev, &weight_dev, &mut out_dev, EPS)?;
    let got = stream.clone_dtoh(&out_dev)?;

    let want = reference(&x, &weight);
    // The GPU sums in a different order, which can move a value across a BF16 rounding boundary.
    let mut exact = 0;
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        let ulps = (i32::from(g.to_bits()) - i32::from(w.to_bits())).abs();
        assert!(ulps <= 1, "element {i}: got {g}, want {w}");
        exact += usize::from(ulps == 0);
    }
    let exact_share = exact as f64 / want.len() as f64;
    assert!(
        exact_share > 0.999,
        "only {exact_share:.5} of elements match exactly"
    );
    Ok(())
}
