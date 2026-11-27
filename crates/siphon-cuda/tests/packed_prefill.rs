//! Ragged packed prefill parity and timing tests.
//!
//! Kernel-level tests verify `attention_post_rows_packed`, `short_conv_residual_rows_packed`,
//! `moe_conv_residual_rows_packed`, `shift_visit_states_packed`, `append_rows_packed`, and
//! `prefill_attention_packed` against their per-slice counterparts on uneven slice lengths
//! (`[1, 17, 64, 129, 256]`, etc.) with nonzero prefix offsets.
//!
//! End-to-end model tests (`#[ignore]`) verify `Model::prefill_packed` against separate
//! `Model::prefill_slot` calls and benchmark `S = 16` & `S = 32` packed prefill latency:
//!
//! ```sh
//! cargo test --release -p siphon-cuda --test packed_prefill -- --nocapture
//! SIPHON_MODEL=<model-dir> cargo test --release -p siphon-cuda --test packed_prefill -- \
//!     --ignored --nocapture --test-threads 1
//! ```

#![cfg(target_os = "linux")]

mod common;

use std::{env, fs, path::PathBuf, sync::Arc, time::Instant};

use anyhow::{Context as _, ensure};
use common::{assert_bf16_close, pseudo_random, pseudo_random_bf16};
use cudarc::driver::{CudaSlice, CudaStream, DevicePtr as _};
use half::bf16;
use safetensors::{Dtype, SafeTensors};
use siphon::{Checkpoint, TextConfig};
use siphon_cuda::{
    Gpu,
    model::{DEFAULT_PREFILL_CHUNK, Model, PackedPrefill, Sampling},
    ops::{
        Products,
        attention::{
            Attention, HEAD_DIM, HEADS, KV_HEADS, KvCache, KvData, KvFormat, MAX_PREFILL_ROWS,
            PREFILL_PACKED_MAX_BLOCKS, PrefillScratch, prefill_blocks,
        },
        conv::{Conv, MOE_CONV_ROWS},
        norm::{AttentionPostRows, AttentionPostRowsPacked, KvConvWeights, Norm},
        spec::{BATCH_STATE_WORDS, PACKED_SLICE_WORDS, PackedRows},
    },
    weights::{WeightFormat, Weights},
};

const HIDDEN: usize = 2304;
const KV_DIMS: usize = KV_HEADS * HEAD_DIM;
const Q_DIMS: usize = HEADS * HEAD_DIM;
const REL_DIM: usize = 16;
const REL_DIMS: usize = HEADS * REL_DIM;
const CONV_TAPS: usize = 4;
const STATE_ROWS: usize = CONV_TAPS - 1;
const TOP_K: usize = 8;
const WINDOW: usize = 1024;
const EXTENT: usize = 512;
const EPS: f32 = 1e-6;
const QK_THRESHOLD: f32 = 22.627_417;
const REL_THRESHOLD: f32 = 8.0;
const LOG_ALPHA: f32 = 0.25;
const LOG_FLOOR: f32 = 128.0;

fn assert_f32_exact(label: &str, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "{label}: length");
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        assert_eq!(
            g.to_bits(),
            w.to_bits(),
            "{label}[{i}]: got {g} ({:#x}), want {w} ({:#x})",
            g.to_bits(),
            w.to_bits()
        );
    }
}

fn kv_ptrs(stream: &CudaStream, kv: &KvCache) -> (u64, u64) {
    match kv.data() {
        KvData::Bf16 { keys, values } => (keys.device_ptr(stream).0, values.device_ptr(stream).0),
        KvData::Fp8 { keys, values } | KvData::Nvfp4 { keys, values } => {
            (keys.device_ptr(stream).0, values.device_ptr(stream).0)
        }
    }
}

type PackedMeta = (usize, CudaSlice<u32>, CudaSlice<u32>, CudaSlice<u32>);

fn build_packed_meta(
    stream: &Arc<CudaStream>,
    specs: &[(usize, usize)],
) -> anyhow::Result<PackedMeta> {
    let mut offset = 0usize;
    let mut slices = Vec::with_capacity(specs.len() * PACKED_SLICE_WORDS);
    let mut attn_blocks = Vec::new();
    let mut moe_tiles = Vec::new();
    for (si, &(len, kv_start)) in specs.iter().enumerate() {
        slices.extend_from_slice(&[
            u32::try_from(offset)?,
            u32::try_from(len)?,
            u32::try_from(kv_start)?,
            u32::try_from(si)?,
        ]);
        for bm in 0..prefill_blocks(len) {
            attn_blocks.push(u32::try_from((bm << 8) | si)?);
        }
        for tile in 0..len.div_ceil(MOE_CONV_ROWS) {
            moe_tiles.push(u32::try_from((tile << 8) | si)?);
        }
        offset += len;
    }
    Ok((
        offset,
        stream.clone_htod(&slices)?,
        stream.clone_htod(&attn_blocks)?,
        stream.clone_htod(&moe_tiles)?,
    ))
}

#[test]
#[expect(clippy::too_many_lines, reason = "tests static/dynamic x sliding/global parity")]
fn attention_post_packed_matches_separate() -> anyhow::Result<()> {
    let gpu = Gpu::new(0)?;
    let stream = gpu.stream();
    let norm = Norm::load(gpu.context())?;
    let conv = Conv::load(gpu.context())?;

    // (len, kv_start) across 5 uneven slices with nonzero prefix offsets.
    let specs: [(usize, usize); 5] = [(1, 0), (17, 33), (64, 511), (129, 1024), (256, 2050)];
    let (m, slices_dev, _, _) = build_packed_meta(stream, &specs)?;
    let taps_w = KV_HEADS * CONV_TAPS;

    let k_raw_dev = stream.clone_htod(&pseudo_random_bf16(m * KV_DIMS, 1, 0.5, 0.0))?;
    let v_raw_dev = stream.clone_htod(&pseudo_random_bf16(m * KV_DIMS, 2, 0.5, 0.0))?;
    let q_raw_dev = stream.clone_htod(&pseudo_random_bf16(m * Q_DIMS, 3, 0.5, 0.0))?;
    let r_raw_dev = stream.clone_htod(&pseudo_random_bf16(m * REL_DIMS, 4, 0.5, 0.0))?;
    let k_dyn_dev = stream.clone_htod(&pseudo_random_bf16(m * taps_w, 5, 0.3, 0.0))?;
    let v_dyn_dev = stream.clone_htod(&pseudo_random_bf16(m * taps_w, 6, 0.3, 0.0))?;
    let k_weight_dev = stream.clone_htod(&pseudo_random_bf16(KV_DIMS * CONV_TAPS, 7, 0.4, 0.0))?;
    let v_weight_dev = stream.clone_htod(&pseudo_random_bf16(KV_DIMS * CONV_TAPS, 8, 0.4, 0.0))?;
    let k_static_dev = stream.clone_htod(&pseudo_random(CONV_TAPS * KV_DIMS, 9, 0.4))?;
    let v_static_dev = stream.clone_htod(&pseudo_random(CONV_TAPS * KV_DIMS, 10, 0.4))?;
    let q_scale_dev = stream.clone_htod(&pseudo_random_bf16(Q_DIMS, 11, 0.5, 1.0))?;
    let r_scale_dev = stream.clone_htod(&pseudo_random_bf16(REL_DIMS, 12, 0.5, 1.0))?;

    for dynamic in [false, true] {
        for log_scale in [false, true] {
            let mut ref_key_states = Vec::with_capacity(specs.len());
            let mut ref_val_states = Vec::with_capacity(specs.len());
            let mut got_key_states = Vec::with_capacity(specs.len());
            let mut got_val_states = Vec::with_capacity(specs.len());
            let mut state_table = Vec::with_capacity(specs.len() * BATCH_STATE_WORDS);

            for si in 0..specs.len() {
                let key_init = pseudo_random(STATE_ROWS * KV_DIMS, 100 + si as u32, 0.5);
                let val_init = pseudo_random(STATE_ROWS * KV_DIMS, 200 + si as u32, 0.5);
                ref_key_states.push(stream.clone_htod(&key_init)?);
                ref_val_states.push(stream.clone_htod(&val_init)?);
                let got_key = stream.clone_htod(&key_init)?;
                let got_val = stream.clone_htod(&val_init)?;
                state_table.extend_from_slice(&[
                    0,
                    0,
                    got_key.device_ptr(stream).0,
                    got_val.device_ptr(stream).0,
                    0,
                    0,
                ]);
                got_key_states.push(got_key);
                got_val_states.push(got_val);
            }
            let states_dev = stream.clone_htod(&state_table)?;

            let mut ref_k = stream.alloc_zeros::<bf16>(m * KV_DIMS)?;
            let mut ref_v = stream.alloc_zeros::<bf16>(m * KV_DIMS)?;
            let mut ref_q = stream.alloc_zeros::<bf16>(m * Q_DIMS)?;
            let mut ref_r = stream.alloc_zeros::<bf16>(m * REL_DIMS)?;
            let mut start_dev = stream.alloc_zeros::<u32>(1)?;

            let mut off = 0usize;
            for (si, &(len, kv_start)) in specs.iter().enumerate() {
                stream.memcpy_htod(&[u32::try_from(kv_start)?], &mut start_dev)?;
                let conv_w = if dynamic {
                    KvConvWeights::Dynamic {
                        k_dynamic: k_dyn_dev.slice(off * taps_w..(off + len) * taps_w),
                        v_dynamic: v_dyn_dev.slice(off * taps_w..(off + len) * taps_w),
                        k_static: &k_static_dev,
                        v_static: &v_static_dev,
                    }
                } else {
                    KvConvWeights::Static {
                        k: &k_weight_dev,
                        v: &v_weight_dev,
                    }
                };
                norm.attention_post_rows(
                    stream,
                    AttentionPostRows {
                        k_raw: k_raw_dev.slice(off * KV_DIMS..(off + len) * KV_DIMS),
                        v_raw: v_raw_dev.slice(off * KV_DIMS..(off + len) * KV_DIMS),
                        conv: conv_w,
                        k_state: &ref_key_states[si],
                        v_state: &ref_val_states[si],
                        q_raw: q_raw_dev.slice(off * Q_DIMS..(off + len) * Q_DIMS),
                        q_scale: &q_scale_dev,
                        r_raw: r_raw_dev.slice(off * REL_DIMS..(off + len) * REL_DIMS),
                        r_scale: &r_scale_dev,
                        k: ref_k.slice_mut(off * KV_DIMS..(off + len) * KV_DIMS),
                        v: ref_v.slice_mut(off * KV_DIMS..(off + len) * KV_DIMS),
                        q: ref_q.slice_mut(off * Q_DIMS..(off + len) * Q_DIMS),
                        r: ref_r.slice_mut(off * REL_DIMS..(off + len) * REL_DIMS),
                        log_scale: log_scale
                            .then_some((start_dev.as_view(), LOG_ALPHA, LOG_FLOOR)),
                        qk_threshold: QK_THRESHOLD,
                        rel_threshold: REL_THRESHOLD,
                        eps: EPS,
                        bias: None,
                        batch: None,
                    },
                    len,
                )?;
                conv.shift_state(
                    stream,
                    &k_raw_dev.slice(off * KV_DIMS..(off + len) * KV_DIMS),
                    &mut ref_key_states[si],
                    KV_DIMS,
                    len,
                )?;
                conv.shift_state(
                    stream,
                    &v_raw_dev.slice(off * KV_DIMS..(off + len) * KV_DIMS),
                    &mut ref_val_states[si],
                    KV_DIMS,
                    len,
                )?;
                off += len;
            }

            let mut got_k = stream.alloc_zeros::<bf16>(m * KV_DIMS)?;
            let mut got_v = stream.alloc_zeros::<bf16>(m * KV_DIMS)?;
            let mut got_q = stream.alloc_zeros::<bf16>(m * Q_DIMS)?;
            let mut got_r = stream.alloc_zeros::<bf16>(m * REL_DIMS)?;
            let packed = PackedRows {
                states: states_dev.as_view(),
                slices: slices_dev.as_view(),
                num_slices: specs.len(),
            };
            let conv_w = if dynamic {
                KvConvWeights::Dynamic {
                    k_dynamic: k_dyn_dev.as_view(),
                    v_dynamic: v_dyn_dev.as_view(),
                    k_static: &k_static_dev,
                    v_static: &v_static_dev,
                }
            } else {
                KvConvWeights::Static {
                    k: &k_weight_dev,
                    v: &v_weight_dev,
                }
            };
            norm.attention_post_rows_packed(
                stream,
                AttentionPostRowsPacked {
                    k_raw: k_raw_dev.as_view(),
                    v_raw: v_raw_dev.as_view(),
                    conv: conv_w,
                    q_raw: q_raw_dev.as_view(),
                    q_scale: &q_scale_dev,
                    r_raw: r_raw_dev.as_view(),
                    r_scale: &r_scale_dev,
                    k: got_k.as_view_mut(),
                    v: got_v.as_view_mut(),
                    q: got_q.as_view_mut(),
                    r: got_r.as_view_mut(),
                    log_scale: log_scale.then_some((LOG_ALPHA, LOG_FLOOR)),
                    qk_threshold: QK_THRESHOLD,
                    rel_threshold: REL_THRESHOLD,
                    eps: EPS,
                    bias: None,
                    packed: &packed,
                },
                m,
            )?;
            conv.shift_visit_states_packed(
                stream,
                [Some(k_raw_dev.as_view()), Some(v_raw_dev.as_view()), None, None],
                &packed,
                m,
            )?;

            let tag = format!("dyn={dynamic}_log={log_scale}");
            assert_bf16_close(&format!("{tag}.k"), &stream.clone_dtoh(&got_k)?, &stream.clone_dtoh(&ref_k)?, 0);
            assert_bf16_close(&format!("{tag}.v"), &stream.clone_dtoh(&got_v)?, &stream.clone_dtoh(&ref_v)?, 0);
            assert_bf16_close(&format!("{tag}.q"), &stream.clone_dtoh(&got_q)?, &stream.clone_dtoh(&ref_q)?, 0);
            assert_bf16_close(&format!("{tag}.r"), &stream.clone_dtoh(&got_r)?, &stream.clone_dtoh(&ref_r)?, 0);
            for si in 0..specs.len() {
                assert_f32_exact(
                    &format!("{tag}.k_state[{si}]"),
                    &stream.clone_dtoh(&got_key_states[si])?,
                    &stream.clone_dtoh(&ref_key_states[si])?,
                );
                assert_f32_exact(
                    &format!("{tag}.v_state[{si}]"),
                    &stream.clone_dtoh(&got_val_states[si])?,
                    &stream.clone_dtoh(&ref_val_states[si])?,
                );
            }
        }
    }
    Ok(())
}

#[test]
#[expect(clippy::too_many_lines, reason = "tests post-attn, post-MLP, and 4-stream shift parity")]
fn conv_residual_packed_matches_separate() -> anyhow::Result<()> {
    let gpu = Gpu::new(0)?;
    let stream = gpu.stream();
    let conv = Conv::load(gpu.context())?;

    // Cover conv window boundaries (1, 2, 3), MoE tile boundaries (8, 9), and uneven lengths.
    let specs: [(usize, usize); 9] = [
        (1, 0),
        (2, 5),
        (3, 10),
        (8, 20),
        (9, 30),
        (17, 40),
        (64, 50),
        (129, 60),
        (256, 70),
    ];
    let (m, slices_dev, _, moe_tiles_dev) = build_packed_meta(stream, &specs)?;

    let k_raw_dev = stream.clone_htod(&pseudo_random_bf16(m * KV_DIMS, 21, 0.5, 0.0))?;
    let v_raw_dev = stream.clone_htod(&pseudo_random_bf16(m * KV_DIMS, 22, 0.5, 0.0))?;
    let proj_dev = stream.clone_htod(&pseudo_random_bf16(m * HIDDEN, 23, 0.5, 0.0))?;
    let hidden_dev = stream.clone_htod(&pseudo_random_bf16(m * HIDDEN, 24, 0.5, 0.0))?;
    let experts_dev = stream.clone_htod(&pseudo_random_bf16(m * TOP_K * HIDDEN, 25, 0.3, 0.0))?;
    let weights_dev = stream.clone_htod(&pseudo_random(m * TOP_K, 26, 0.5))?;
    let post_attn_w = stream.clone_htod(&pseudo_random_bf16(HIDDEN * CONV_TAPS, 27, 0.4, 0.0))?;
    let post_mlp_w = stream.clone_htod(&pseudo_random_bf16(HIDDEN * CONV_TAPS, 28, 0.4, 0.0))?;

    for with_hidden in [false, true] {
        let mut ref_states: Vec<[CudaSlice<f32>; 4]> = Vec::with_capacity(specs.len());
        let mut got_states: Vec<[CudaSlice<f32>; 4]> = Vec::with_capacity(specs.len());
        let mut table = Vec::with_capacity(specs.len() * BATCH_STATE_WORDS);

        for si in 0..specs.len() {
            let s0 = pseudo_random(STATE_ROWS * KV_DIMS, 300 + si as u32, 0.5);
            let s1 = pseudo_random(STATE_ROWS * KV_DIMS, 400 + si as u32, 0.5);
            let s2 = pseudo_random(STATE_ROWS * HIDDEN, 500 + si as u32, 0.5);
            let s3 = pseudo_random(STATE_ROWS * HIDDEN, 600 + si as u32, 0.5);
            ref_states.push([
                stream.clone_htod(&s0)?,
                stream.clone_htod(&s1)?,
                stream.clone_htod(&s2)?,
                stream.clone_htod(&s3)?,
            ]);
            let g = [
                stream.clone_htod(&s0)?,
                stream.clone_htod(&s1)?,
                stream.clone_htod(&s2)?,
                stream.clone_htod(&s3)?,
            ];
            table.extend_from_slice(&[
                0,
                0,
                g[0].device_ptr(stream).0,
                g[1].device_ptr(stream).0,
                g[2].device_ptr(stream).0,
                g[3].device_ptr(stream).0,
            ]);
            got_states.push(g);
        }
        let states_dev = stream.clone_htod(&table)?;

        let mut ref_res = stream.alloc_zeros::<f32>(m * HIDDEN)?;
        let mut ref_comb = stream.alloc_zeros::<bf16>(m * HIDDEN)?;
        let mut ref_out = stream.alloc_zeros::<bf16>(m * HIDDEN)?;
        let mut off = 0usize;
        for (si, &(len, _)) in specs.iter().enumerate() {
            conv.shift_state(
                stream,
                &k_raw_dev.slice(off * KV_DIMS..(off + len) * KV_DIMS),
                &mut ref_states[si][0],
                KV_DIMS,
                len,
            )?;
            conv.shift_state(
                stream,
                &v_raw_dev.slice(off * KV_DIMS..(off + len) * KV_DIMS),
                &mut ref_states[si][1],
                KV_DIMS,
                len,
            )?;
            let h_slice =
                with_hidden.then(|| hidden_dev.slice(off * HIDDEN..(off + len) * HIDDEN));
            conv.short_conv_residual_rows(
                stream,
                &proj_dev.slice(off * HIDDEN..(off + len) * HIDDEN),
                &post_attn_w,
                &mut ref_states[si][2],
                h_slice.as_ref().map(|v| v as &dyn siphon_cuda::DeviceIn<bf16>),
                &mut ref_res.slice_mut(off * HIDDEN..(off + len) * HIDDEN),
                len,
            )?;
            conv.moe_conv_residual_rows(
                stream,
                &experts_dev.slice(off * TOP_K * HIDDEN..(off + len) * TOP_K * HIDDEN),
                &weights_dev.slice(off * TOP_K..(off + len) * TOP_K),
                &post_mlp_w,
                &mut ref_states[si][3],
                &ref_res.slice(off * HIDDEN..(off + len) * HIDDEN),
                &mut ref_comb.slice_mut(off * HIDDEN..(off + len) * HIDDEN),
                &mut ref_out.slice_mut(off * HIDDEN..(off + len) * HIDDEN),
                len,
            )?;
            off += len;
        }

        let mut got_res = stream.alloc_zeros::<f32>(m * HIDDEN)?;
        let mut got_comb = stream.alloc_zeros::<bf16>(m * HIDDEN)?;
        let mut got_out = stream.alloc_zeros::<bf16>(m * HIDDEN)?;
        let packed = PackedRows {
            states: states_dev.as_view(),
            slices: slices_dev.as_view(),
            num_slices: specs.len(),
        };
        let hidden_view = with_hidden.then(|| hidden_dev.as_view());
        conv.short_conv_residual_rows_packed(
            stream,
            &proj_dev,
            &post_attn_w,
            &packed,
            hidden_view.as_ref(),
            &mut got_res,
            m,
        )?;
        conv.moe_conv_residual_rows_packed(
            stream,
            &experts_dev,
            &weights_dev,
            &post_mlp_w,
            &packed,
            &moe_tiles_dev.as_view(),
            &got_res,
            &mut got_comb,
            &mut got_out,
            m,
        )?;
        conv.shift_visit_states_packed(
            stream,
            [
                Some(k_raw_dev.as_view()),
                Some(v_raw_dev.as_view()),
                Some(proj_dev.as_view()),
                Some(got_comb.as_view()),
            ],
            &packed,
            m,
        )?;

        assert_f32_exact(
            &format!("hidden={with_hidden}.residual"),
            &stream.clone_dtoh(&got_res)?,
            &stream.clone_dtoh(&ref_res)?,
        );
        assert_bf16_close(
            &format!("hidden={with_hidden}.out"),
            &stream.clone_dtoh(&got_out)?,
            &stream.clone_dtoh(&ref_out)?,
            0,
        );
        for si in 0..specs.len() {
            for st in 0..4 {
                assert_f32_exact(
                    &format!("hidden={with_hidden}.state[{si}][{st}]"),
                    &stream.clone_dtoh(&got_states[si][st])?,
                    &stream.clone_dtoh(&ref_states[si][st])?,
                );
            }
        }
    }
    Ok(())
}

fn check_kv_equal(
    stream: &Arc<CudaStream>,
    label: &str,
    got: &KvCache,
    want: &KvCache,
) -> anyhow::Result<()> {
    match (got.data(), want.data()) {
        (
            KvData::Bf16 { keys: gk, values: gv },
            KvData::Bf16 { keys: wk, values: wv },
        ) => {
            assert_bf16_close(&format!("{label}.k_cache"), &stream.clone_dtoh(gk)?, &stream.clone_dtoh(wk)?, 0);
            assert_bf16_close(&format!("{label}.v_cache"), &stream.clone_dtoh(gv)?, &stream.clone_dtoh(wv)?, 0);
        }
        (
            KvData::Fp8 { keys: gk, values: gv },
            KvData::Fp8 { keys: wk, values: wv },
        )
        | (
            KvData::Nvfp4 { keys: gk, values: gv },
            KvData::Nvfp4 { keys: wk, values: wv },
        ) => {
            assert_eq!(stream.clone_dtoh(gk)?, stream.clone_dtoh(wk)?, "{label}.k_cache");
            assert_eq!(stream.clone_dtoh(gv)?, stream.clone_dtoh(wv)?, "{label}.v_cache");
        }
        _ => anyhow::bail!("{label}: mismatched KvCache formats"),
    }
    Ok(())
}

#[expect(clippy::too_many_lines, reason = "sets up prefixes and compares packed vs separate attention")]
fn check_attention_packed(
    label: &str,
    make_cache: impl Fn(&Arc<CudaStream>) -> anyhow::Result<KvCache>,
    specs: &[(usize, usize)],
    products: Products,
    max_ulps: u32,
) -> anyhow::Result<()> {
    let gpu = Gpu::new(0)?;
    let stream = gpu.stream();
    let attention = Attention::load(gpu.context())?;
    let mut scratch = PrefillScratch::new(stream, MAX_PREFILL_ROWS)?;
    let mut counters = stream.alloc_zeros::<u32>(PREFILL_PACKED_MAX_BLOCKS * KV_HEADS)?;

    let (m, slices_dev, attn_blocks_dev, _) = build_packed_meta(stream, specs)?;

    let mut ref_caches = Vec::with_capacity(specs.len());
    let mut got_caches = Vec::with_capacity(specs.len());
    let mut state_table = Vec::with_capacity(specs.len() * BATCH_STATE_WORDS);

    // Pre-populate each slice's KV cache with `kv_start` prefix tokens so both sliding ring
    // wraparound and global prefix reads are exercised.
    let mut pos_dev = stream.alloc_zeros::<u32>(1)?;
    for (si, &(_, kv_start)) in specs.iter().enumerate() {
        let mut rc = make_cache(stream)?;
        let mut gc = make_cache(stream)?;
        if kv_start > 0 {
            let pk = stream.clone_htod(&pseudo_random_bf16(kv_start * KV_DIMS, 700 + si as u32, 0.5, 0.0))?;
            let pv = stream.clone_htod(&pseudo_random_bf16(kv_start * KV_DIMS, 800 + si as u32, 0.5, 0.0))?;
            let pq = stream.clone_htod(&pseudo_random_bf16(kv_start * Q_DIMS, 850 + si as u32, 0.5, 0.0))?;
            let pb = stream.clone_htod(&pseudo_random_bf16(kv_start * HEADS * EXTENT, 860 + si as u32, 0.5, 0.0))?;
            let ps = stream.clone_htod(&pseudo_random_bf16(kv_start * HEADS, 870 + si as u32, 0.5, 0.0))?;
            let mut po = stream.alloc_zeros::<bf16>(kv_start * Q_DIMS)?;
            stream.memcpy_htod(&[0u32], &mut pos_dev)?;
            attention.prefill_attention(
                stream,
                &pq,
                &pk,
                &pv,
                &mut rc,
                &pos_dev,
                kv_start,
                &pb,
                &ps,
                &mut scratch,
                &mut po,
                products,
            )?;
            attention.prefill_attention(
                stream,
                &pq,
                &pk,
                &pv,
                &mut gc,
                &pos_dev,
                kv_start,
                &pb,
                &ps,
                &mut scratch,
                &mut po,
                products,
            )?;
        }
        let (kp, vp) = kv_ptrs(stream, &gc);
        state_table.extend_from_slice(&[kp, vp, 0, 0, 0, 0]);
        ref_caches.push(rc);
        got_caches.push(gc);
    }
    let states_dev = stream.clone_htod(&state_table)?;

    let q_dev = stream.clone_htod(&pseudo_random_bf16(m * Q_DIMS, 31, 0.5, 0.0))?;
    let k_dev = stream.clone_htod(&pseudo_random_bf16(m * KV_DIMS, 32, 0.5, 0.0))?;
    let v_dev = stream.clone_htod(&pseudo_random_bf16(m * KV_DIMS, 33, 0.5, 0.0))?;
    let bias_dev = stream.clone_htod(&pseudo_random_bf16(m * HEADS * EXTENT, 34, 1.0, 0.0))?;
    let sinks_dev = stream.clone_htod(&pseudo_random_bf16(m * HEADS, 35, 1.5, 0.0))?;

    let mut ref_out = stream.alloc_zeros::<bf16>(m * Q_DIMS)?;
    let mut off = 0usize;
    for (si, &(len, kv_start)) in specs.iter().enumerate() {
        stream.memcpy_htod(&[u32::try_from(kv_start)?], &mut pos_dev)?;
        attention.prefill_attention(
            stream,
            &q_dev.slice(off * Q_DIMS..(off + len) * Q_DIMS),
            &k_dev.slice(off * KV_DIMS..(off + len) * KV_DIMS),
            &v_dev.slice(off * KV_DIMS..(off + len) * KV_DIMS),
            &mut ref_caches[si],
            &pos_dev,
            len,
            &bias_dev.slice(off * HEADS * EXTENT..(off + len) * HEADS * EXTENT),
            &sinks_dev.slice(off * HEADS..(off + len) * HEADS),
            &mut scratch,
            &mut ref_out.slice_mut(off * Q_DIMS..(off + len) * Q_DIMS),
            products,
        )?;
        off += len;
    }

    let mut got_out = stream.alloc_zeros::<bf16>(m * Q_DIMS)?;
    let packed = PackedRows {
        states: states_dev.as_view(),
        slices: slices_dev.as_view(),
        num_slices: specs.len(),
    };
    attention.prefill_attention_packed(
        stream,
        &q_dev,
        &k_dev,
        &v_dev,
        &got_caches[0],
        &packed,
        &attn_blocks_dev.as_view(),
        m,
        &bias_dev,
        &sinks_dev,
        &mut scratch,
        &mut counters,
        &mut got_out,
        products,
    )?;

    // Verify all counters were returned to zero.
    let counter_vals = stream.clone_dtoh(&counters)?;
    ensure!(
        counter_vals.iter().all(|&c| c == 0),
        "{label}: packed attention left nonzero counters"
    );

    // KV caches must match bit-for-bit on every slice.
    for si in 0..specs.len() {
        check_kv_equal(stream, &format!("{label}[{si}]"), &got_caches[si], &ref_caches[si])?;
    }

    let got = stream.clone_dtoh(&got_out)?;
    let want = stream.clone_dtoh(&ref_out)?;
    if max_ulps == 0 {
        assert_bf16_close(label, &got, &want, 0);
    } else {
        // Multi-slice runs pack all S slices onto one persistent grid (different split
        // boundaries than separate single-slice calls, and len=1 uses tensor-core MMA
        // instead of decode_attention). Check exact BF16 parity via tight absolute +
        // relative tolerance (matching `tests/attention.rs`), and report ULP stats.
        let mut max_abs = 0.0f32;
        let mut worst_ulps = 0u32;
        let mut differing = 0usize;
        for (i, (&g, &w)) in got.iter().zip(&want).enumerate() {
            let gf = g.to_f32();
            let wf = w.to_f32();
            let err = (gf - wf).abs();
            let ulps = common::bf16_ulps(g, w);
            max_abs = max_abs.max(err);
            worst_ulps = worst_ulps.max(ulps);
            differing += usize::from(ulps > 0);
            ensure!(
                err <= 2e-3 + 2e-2 * wf.abs(),
                "{label}[{i}]: got {g} ({gf}), want {w} ({wf}), err={err:e} ({ulps} ulps)"
            );
        }
        eprintln!(
            "{label}: {differing} of {} elements differ (max_abs={max_abs:e}, worst_ulps={worst_ulps})",
            want.len()
        );
    }
    Ok(())
}

#[test]
fn prefill_attention_packed_single_slice_bit_exact() -> anyhow::Result<()> {
    for len in [17, 64, 129, 256] {
        check_attention_packed(
            &format!("single_sliding_bf16_len{len}"),
            |s| KvCache::sliding(s, WINDOW),
            &[(len, 1020)],
            Products::Bf16,
            0,
        )?;
        check_attention_packed(
            &format!("single_global_fp8_len{len}"),
            |s| KvCache::global_with_format(s, 2048, KvFormat::Fp8),
            &[(len, 1020)],
            Products::Bf16,
            0,
        )?;
    }
    Ok(())
}

#[test]
fn prefill_attention_packed_uneven_matches_separate_sliding_bf16() -> anyhow::Result<()> {
    let specs = [(1, 0), (17, 37), (64, 512), (129, 1020), (256, 1500)];
    check_attention_packed(
        "uneven_sliding_bf16",
        |s| KvCache::sliding(s, WINDOW),
        &specs,
        Products::Bf16,
        1,
    )
}

#[test]
fn prefill_attention_packed_uneven_matches_separate_sliding_fp8() -> anyhow::Result<()> {
    let specs = [(1, 0), (17, 37), (64, 512), (129, 1020), (256, 1500)];
    check_attention_packed(
        "uneven_sliding_fp8",
        |s| KvCache::sliding_ring_with_format(s, WINDOW, WINDOW, KvFormat::Fp8),
        &specs,
        Products::Bf16,
        1,
    )
}

#[test]
fn prefill_attention_packed_uneven_matches_separate_global_bf16() -> anyhow::Result<()> {
    let specs = [(1, 0), (17, 37), (64, 512), (129, 1020), (256, 1500)];
    check_attention_packed(
        "uneven_global_bf16",
        |s| KvCache::global(s, 2048),
        &specs,
        Products::Bf16,
        1,
    )
}

#[test]
fn prefill_attention_packed_uneven_matches_separate_global_fp8() -> anyhow::Result<()> {
    let specs = [(1, 0), (17, 37), (64, 512), (129, 1020), (256, 1500)];
    check_attention_packed(
        "uneven_global_fp8",
        |s| KvCache::global_with_format(s, 2048, KvFormat::Fp8),
        &specs,
        Products::Bf16,
        1,
    )
}

#[test]
fn prefill_attention_packed_uneven_matches_separate_global_fp8_mma() -> anyhow::Result<()> {
    // Exclude len=1 when comparing FP8 MMA against single-row decode_attention (which keeps Q
    // unquantized); test len=1..256 with the non-MMA path above and len=2..256 here.
    let specs = [(2, 0), (17, 37), (64, 512), (129, 1020), (256, 1500)];
    check_attention_packed(
        "uneven_global_fp8_mma",
        |s| KvCache::global_with_format(s, 2048, KvFormat::Fp8),
        &specs,
        Products::Fp8,
        1,
    )
}

#[test]
fn prefill_attention_packed_uneven_matches_separate_global_fp4() -> anyhow::Result<()> {
    let specs = [(1, 0), (17, 37), (64, 512), (129, 1020), (256, 1500)];
    check_attention_packed(
        "uneven_global_fp4",
        |s| KvCache::global_with_format(s, 2048, KvFormat::Nvfp4),
        &specs,
        Products::Bf16,
        1,
    )
}

// ---------------------------------------------------------------------------
// End-to-end Model::prefill_packed parity and S=16 / S=32 latency benchmarks.
// ---------------------------------------------------------------------------

fn capture_tokens() -> anyhow::Result<Vec<u32>> {
    let path = env::var_os("SIPHON_CAPTURE").map_or_else(
        || {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../reference/out/anyhow-chunked.safetensors")
        },
        PathBuf::from,
    );
    let bytes = fs::read(&path).with_context(|| format!("reading {}", path.display()))?;
    let capture = SafeTensors::deserialize(&bytes)?;
    let view = capture.tensor("tokens")?;
    ensure!(view.dtype() == Dtype::I32, "tokens are {:?}", view.dtype());
    view.data()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| Ok(u32::try_from(i32::from_le_bytes(*b))?))
        .collect()
}

fn load_model(capacity: usize, kv: KvFormat, weights: WeightFormat) -> anyhow::Result<(Gpu, Model)> {
    let model_dir = PathBuf::from(env::var_os("SIPHON_MODEL").context("set SIPHON_MODEL")?);
    let gpu = Gpu::new(0)?;
    let config = TextConfig::load(&model_dir)?;
    let passes = config.loop_config.num_passes;
    let checkpoint = Checkpoint::open(&model_dir)?;
    let loaded = Weights::load(&checkpoint, &config, gpu.stream())?;
    let model = Model::with_formats(&gpu, config, loaded, passes, capacity, kv, weights)?;
    Ok((gpu, model))
}

/// End-to-end parity test: `Model::prefill_packed` with uneven slice lengths
/// `[1, 17, 64, 129, 256]` AND nonzero prefix offsets `[0, 31, 128, 1020, 513]` matches
/// separate `Model::prefill_slot` calls on first tokens and subsequent batched decode logits.
/// Also verifies single-slice (`S=1`) `len=1` and `len=64` at prefix offset `1020` match
/// `prefill_slot` bit-for-bit (`max_diff == 0.0`).
#[test]
#[ignore = "needs the checkpoint and a GPU"]
#[expect(clippy::too_many_lines, reason = "end-to-end ragged packed prefill parity test")]
fn model_prefill_packed_uneven_with_prefixes_matches_separate() -> anyhow::Result<()> {
    let (_gpu, mut model) = load_model(4096, KvFormat::Fp8, WeightFormat::Fp8)?;
    let capture = capture_tokens()?;
    // (capture_offset, prefix_len, suffix_len)
    let specs: [(usize, usize, usize); 5] = [
        (0, 0, 1),
        (200, 31, 17),
        (600, 128, 64),
        (1000, 1020, 129),
        (2500, 513, 256),
    ];
    let n = specs.len();
    model.add_slots(n - 1)?;
    let vocab = model.config().unpadded_vocab_size;

    // 1. Reference: separate `prefill_slot` for prefix then suffix, then 1 batched decode step.
    let mut want_first = Vec::with_capacity(n);
    let mut want_logits = Vec::with_capacity(n);
    for (slot, &(base, pref, suff)) in specs.iter().enumerate() {
        model.reset_slot(slot)?;
        if pref > 0 {
            let _ = model.prefill_slot(slot, &capture[base..base + pref], DEFAULT_PREFILL_CHUNK)?;
        }
        let tok = model.prefill_slot(
            slot,
            &capture[base + pref..base + pref + suff],
            DEFAULT_PREFILL_CHUNK,
        )?;
        let _ = model.step_batch(&[(slot, tok)])?;
        want_first.push(tok);
        want_logits.push(model.batch_logits(1)?);
    }

    // 2. Single-slice packed parity (S=1) at both len=1 (prefix=0) and len=129 (prefix=1020):
    //    must match `prefill_slot` bit-for-bit (`max_diff == 0.0`).
    for idx in [0usize, 3usize] {
        let (base, pref, suff) = specs[idx];
        model.reset_slot(0)?;
        if pref > 0 {
            let _ = model.prefill_slot(0, &capture[base..base + pref], DEFAULT_PREFILL_CHUNK)?;
        }
        let one = model.prefill_packed(
            &[PackedPrefill {
                slot: 0,
                tokens: &capture[base + pref..base + pref + suff],
                images: Vec::new(),
                need_first_token: true,
                sampling: Sampling::GREEDY,
            }],
            DEFAULT_PREFILL_CHUNK,
        )?;
        let tok = one[0].context("missing first token for S=1 packed slice")?;
        let _ = model.step_batch(&[(0, tok)])?;
        let got_logits = model.batch_logits(1)?;
        let max_diff = got_logits
            .iter()
            .zip(&want_logits[idx])
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        println!(
            "S=1 (prefix={pref}, len={suff}): want={} got={tok} max_diff={max_diff:.6}",
            want_first[idx]
        );
        ensure!(
            tok == want_first[idx] && max_diff == 0.0,
            "S=1 (prefix={pref}, len={suff}) diverged: max_diff={max_diff}"
        );
    }

    // 3. Packed: prefill prefixes, then run all 5 uneven suffix slices in ONE `prefill_packed`.
    for (slot, &(base, pref, _)) in specs.iter().enumerate() {
        model.reset_slot(slot)?;
        if pref > 0 {
            let _ = model.prefill_slot(slot, &capture[base..base + pref], DEFAULT_PREFILL_CHUNK)?;
        }
    }
    let slices: Vec<PackedPrefill<'_>> = specs
        .iter()
        .enumerate()
        .map(|(slot, &(base, pref, suff))| PackedPrefill {
            slot,
            tokens: &capture[base + pref..base + pref + suff],
            images: Vec::new(),
            need_first_token: true,
            sampling: Sampling::GREEDY,
        })
        .collect();
    let firsts = model.prefill_packed(&slices, DEFAULT_PREFILL_CHUNK)?;
    let toks: Vec<u32> = firsts
        .into_iter()
        .enumerate()
        .map(|(i, t)| t.with_context(|| format!("missing first token for slice {i}")))
        .collect::<anyhow::Result<_>>()?;
    let rows: Vec<(usize, u32)> = toks.iter().copied().enumerate().collect();
    let _ = model.step_batch(&rows)?;
    let got_logits = model.batch_logits(n)?;
    let kl_and_cosine = |want_row: &[f32], got_row: &[f32]| -> (f64, f64) {
        let want_max = want_row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let got_max = got_row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let want_logz = want_row
            .iter()
            .map(|&v| f64::from(v - want_max).exp())
            .sum::<f64>()
            .ln()
            + f64::from(want_max);
        let got_logz = got_row
            .iter()
            .map(|&v| f64::from(v - got_max).exp())
            .sum::<f64>()
            .ln()
            + f64::from(got_max);
        let (mut kl, mut dot, mut want_sq, mut got_sq) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        for (&want_elem, &got_elem) in want_row.iter().zip(got_row) {
            let want_f64 = f64::from(want_elem);
            let got_f64 = f64::from(got_elem);
            let want_lp = want_f64 - want_logz;
            let got_lp = got_f64 - got_logz;
            kl += want_lp.exp() * (want_lp - got_lp);
            dot += want_f64 * got_f64;
            want_sq += want_f64 * want_f64;
            got_sq += got_f64 * got_f64;
        }
        (kl, dot / (want_sq.sqrt() * got_sq.sqrt()))
    };
    for i in 0..n {
        let got_row = &got_logits[i * vocab..(i + 1) * vocab];
        let want_row = &want_logits[i];
        let mut max_diff = 0.0f32;
        let mut sum_diff = 0.0f64;
        for (&g, &w) in got_row.iter().zip(want_row) {
            let d = (g - w).abs();
            max_diff = max_diff.max(d);
            sum_diff += f64::from(d);
        }
        let mean_diff = sum_diff / vocab as f64;
        let (kl, cosine) = kl_and_cosine(want_row, got_row);
        println!(
            "slice {i} (prefix={:4}, len={:3}): want={} got={} KL={kl:.5} cosine={cosine:.5} max_diff={max_diff:.4} mean_diff={mean_diff:.5}",
            specs[i].1, specs[i].2, want_first[i], toks[i]
        );
        ensure!(
            toks[i] == want_first[i],
            "packed slice {i} first token diverged: got {}, want {}",
            toks[i],
            want_first[i]
        );
        // Packed prefill runs BF16 GEMM / expert_matmul / TC router at the combined row count
        // (`m = 467`) while separate prefill runs each slice at its own length (`m = 1..256`).
        // Bound the output distribution by KL divergence and cosine similarity so constant
        // low-probability tail shifts do not false-fail exact top-1 matches.
        ensure!(
            kl < 0.01 && cosine > 0.97,
            "packed slice {i} distribution diverged: KL={kl:.5} cosine={cosine:.5} \
             (max_diff={max_diff:.4} mean_diff={mean_diff:.5})"
        );
    }
    Ok(())
}

/// Timing benchmark comparing `Model::prefill_packed` against sequential `Model::prefill_slot`
/// calls at `S = 16` and `S = 32` concurrent short prompts (both uniform 64-token prompts and
/// ragged uneven prompt lengths).
#[test]
#[ignore = "timing benchmark; needs the checkpoint and a GPU"]
fn model_prefill_packed_timing_s16_s32() -> anyhow::Result<()> {
    let (gpu, mut model) = load_model(2048, KvFormat::Fp8, WeightFormat::Fp8)?;
    let capture = capture_tokens()?;
    model.add_slots(31)?;

    let cases: [(&str, Vec<(usize, usize)>); 4] = [
        (
            "S=16 x 64 (1024 tok)",
            (0..16).map(|i| (i * 100, 64)).collect(),
        ),
        (
            "S=32 x 64 (2048 tok)",
            (0..32).map(|i| (i * 100, 64)).collect(),
        ),
        (
            "S=16 ragged (1024 tok)",
            [16, 32, 48, 64, 80, 96, 112, 128, 17, 31, 47, 57, 65, 73, 79, 79]
                .into_iter()
                .enumerate()
                .map(|(i, len)| (i * 120, len))
                .collect(),
        ),
        (
            "S=32 ragged (2048 tok)",
            (0..32)
                .map(|i| {
                    let lens = [17, 31, 48, 64, 79, 96, 113, 64];
                    (i * 100, lens[i % lens.len()])
                })
                .collect(),
        ),
    ];

    for (label, specs) in &cases {
        let total_toks: usize = specs.iter().map(|&(_, l)| l).sum();
        let slices: Vec<PackedPrefill<'_>> = specs
            .iter()
            .enumerate()
            .map(|(slot, &(base, len))| PackedPrefill {
                slot,
                tokens: &capture[base..base + len],
                images: Vec::new(),
                need_first_token: true,
                sampling: Sampling::GREEDY,
            })
            .collect();

        // Warmup packed.
        for slot in 0..specs.len() {
            model.reset_slot(slot)?;
        }
        let _ = model.prefill_packed(&slices, DEFAULT_PREFILL_CHUNK)?;
        gpu.stream().synchronize()?;

        let mut packed_ms = Vec::new();
        for _ in 0..5 {
            for slot in 0..specs.len() {
                model.reset_slot(slot)?;
            }
            gpu.stream().synchronize()?;
            let t0 = Instant::now();
            let _ = model.prefill_packed(&slices, DEFAULT_PREFILL_CHUNK)?;
            gpu.stream().synchronize()?;
            packed_ms.push(t0.elapsed().as_secs_f64() * 1e3);
        }
        packed_ms.sort_by(f64::total_cmp);

        let mut seq_ms = Vec::new();
        for _ in 0..3 {
            for slot in 0..specs.len() {
                model.reset_slot(slot)?;
            }
            gpu.stream().synchronize()?;
            let t0 = Instant::now();
            for (slot, &(base, len)) in specs.iter().enumerate() {
                let _ = model.prefill_slot(
                    slot,
                    &capture[base..base + len],
                    DEFAULT_PREFILL_CHUNK,
                )?;
            }
            gpu.stream().synchronize()?;
            seq_ms.push(t0.elapsed().as_secs_f64() * 1e3);
        }
        seq_ms.sort_by(f64::total_cmp);

        let p_med = packed_ms[packed_ms.len() / 2];
        let s_med = seq_ms[seq_ms.len() / 2];
        println!(
            "{label:<24}: packed={p_med:6.2} ms ({:8.0} tok/s)  seq={s_med:6.2} ms  speedup={:.2}x",
            total_toks as f64 / (p_med * 1e-3),
            s_med / p_med
        );
    }
    Ok(())
}
