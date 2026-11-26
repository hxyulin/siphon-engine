//! Times the vision tower on synthetic images and reports its device memory.
//!
//! usage: vision <model-dir> [--iters N] [--sizes WxH,WxH,...]
//!        vision <model-dir> --dump <capture-dir> <out-dir>
//!
//! `--dump` instead encodes the six images of `reference/fixtures/vision/` from the
//! patches in `<capture-dir>/<name>.safetensors` (`reference/vision_capture.py capture`)
//! and writes the embeddings to `<out-dir>/<name>.safetensors` as F32 `bf16`,
//! `[tokens, 2304]`, for `reference/vision_text.py`.
//!
//! Each image is padded to a multiple of 40 pixels, as Transformers' image processor does,
//! so 1920x1080 is 96 x 54 patches. Times are the median of `--iters` encodes (default 20) after one warm-up,
//! each measured from enqueue to stream synchronize.

#[cfg(target_os = "linux")]
fn main() -> anyhow::Result<()> {
    linux::main()
}

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("vision needs the CUDA backend, which builds only on Linux");
    std::process::exit(1);
}

#[cfg(target_os = "linux")]
mod linux {
    use std::{
        env, fs,
        path::{Path, PathBuf},
        time::Instant,
    };

    use anyhow::Context as _;
    use safetensors::{Dtype, SafeTensors, tensor::TensorView};
    use siphon::{Checkpoint, memory};
    use siphon_cuda::{
        Gpu,
        vision::{PreparedImage, VisionFormat, VisionTower},
    };

    /// A smooth synthetic image of `w` x `h` pixels, padded to multiples of 40, as patch
    /// rows.
    fn image(width: usize, height: usize) -> PreparedImage {
        let (gh, gw) = (height.div_ceil(40) * 2, width.div_ceil(40) * 2);
        let patches = gh * gw;
        let mut values = Vec::with_capacity(patches * 1200);
        for p in 0..patches {
            let token = p / 4;
            let row = token / (gw / 2) * 2 + (p / 2) % 2;
            let col = token % (gw / 2) * 2 + p % 2;
            for py in 0..20 {
                for px in 0..20 {
                    let (y, x) = (row * 20 + py, col * 20 + px);
                    for c in 0..3 {
                        let phase = c as f32 * 2.1;
                        let v = (x as f32 * 0.013 + phase).sin() * (y as f32 * 0.021 - phase).cos();
                        let inside = y < height && x < width;
                        values.push(if inside { v * 0.9 } else { -1.0 });
                    }
                }
            }
        }
        PreparedImage {
            grid: [1, gh, gw],
            patches: values,
            hash: [0; 32],
        }
    }

    fn median(mut v: Vec<f64>) -> f64 {
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    }

    fn gib(bytes: usize) -> String {
        format!("{:.3} GiB", bytes as f64 / f64::from(1u32 << 30))
    }

    /// The fixture images, by name.
    const FIXTURES: [&str; 6] = ["tiny", "alpha", "tall", "square", "screenshot", "huge"];

    /// The patches of fixture `name` and its grid.
    fn fixture(capture: &Path, name: &str) -> anyhow::Result<PreparedImage> {
        let json = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join(format!("../../reference/fixtures/vision/{name}.json"));
        let meta: serde_json::Value = serde_json::from_str(&fs::read_to_string(&json)?)?;
        let grid: Vec<usize> = meta["grid"]
            .as_array()
            .context("no grid")?
            .iter()
            .map(|v| v.as_u64().map(|v| v as usize).context("grid entry"))
            .collect::<anyhow::Result<_>>()?;
        let bytes = fs::read(capture.join(format!("{name}.safetensors")))?;
        let tensors = SafeTensors::deserialize(&bytes)?;
        let view = tensors.tensor("patches")?;
        anyhow::ensure!(view.dtype() == Dtype::F32, "patches are not F32");
        let patches = view
            .data()
            .as_chunks()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect();
        Ok(PreparedImage {
            grid: [grid[0], grid[1], grid[2]],
            patches,
            hash: [0; 32],
        })
    }

    fn dump(model: &Path, capture: &Path, out: &Path) -> anyhow::Result<()> {
        let checkpoint = Checkpoint::open(model)?;
        let gpu = Gpu::new(0)?;
        fs::create_dir_all(out)?;
        let mut tower = VisionTower::load(&gpu, &checkpoint, VisionFormat::Bf16)?;
        for name in FIXTURES {
            let image = fixture(capture, name)?;
            let embeddings = gpu.stream().clone_dtoh(&tower.encode(&image)?)?;
            let bytes: Vec<u8> = embeddings
                .iter()
                .flat_map(|x| x.to_f32().to_le_bytes())
                .collect();
            let view =
                TensorView::new(Dtype::F32, vec![image.tokens(), memory::VISION_OUT], &bytes)?;
            let path = out.join(format!("{name}.safetensors"));
            fs::write(&path, safetensors::serialize([("bf16", view)], None)?)?;
            println!("wrote {}", path.display());
        }
        Ok(())
    }

    pub fn main() -> anyhow::Result<()> {
        let mut args = env::args().skip(1);
        let model = PathBuf::from(args.next().context("usage: vision <model-dir>")?);
        if args.len() == 3 {
            let mut rest = args;
            anyhow::ensure!(
                rest.next().as_deref() == Some("--dump"),
                "unknown arguments"
            );
            let capture = PathBuf::from(rest.next().context("--dump needs a capture dir")?);
            let out = PathBuf::from(rest.next().context("--dump needs an out dir")?);
            return dump(&model, &capture, &out);
        }
        let mut iters = 20;
        let mut sizes = vec![(512, 512), (1024, 1024), (1920, 1080), (2048, 2048)];
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--iters" => iters = args.next().context("--iters needs a value")?.parse()?,
                "--sizes" => {
                    sizes = args
                        .next()
                        .context("--sizes needs a value")?
                        .split(',')
                        .map(|s| {
                            let (w, h) = s.split_once('x').context("sizes are WxH")?;
                            Ok((w.parse()?, h.parse()?))
                        })
                        .collect::<anyhow::Result<_>>()?;
                }
                other => anyhow::bail!("unknown argument {other}"),
            }
        }
        let checkpoint = Checkpoint::open(&model)?;
        let gpu = Gpu::new(0)?;
        let images: Vec<_> = sizes.iter().map(|&(w, h)| image(w, h)).collect();
        let format = VisionFormat::Bf16;
        let base = gpu.memory()?.pool_used;
        let mut tower = VisionTower::load(&gpu, &checkpoint, format)?;
        let weights = gpu.memory()?.pool_used - base;
        println!(
            "weights {} (planned {})",
            gib(weights),
            gib(memory::vision_weight_bytes(format))
        );
        for (&(w, h), img) in sizes.iter().zip(&images) {
            let patches = img.patches.len() / 1200;
            tower.release();
            gpu.stream().synchronize()?;
            let before = gpu.memory()?.pool_used;
            tower.reserve(patches)?;
            let out = tower.encode(img)?;
            gpu.stream().synchronize()?;
            let used = gpu.memory()?.pool_used - before;
            drop(out);
            let mut times = Vec::with_capacity(iters);
            for _ in 0..iters {
                let t = Instant::now();
                let o = tower.encode(img)?;
                gpu.stream().synchronize()?;
                times.push(t.elapsed().as_secs_f64() * 1e3);
                drop(o);
            }
            println!(
                "{w}x{h}: {patches} patches, {} tokens, {:.2} ms, buffers {} (planned {})",
                img.tokens(),
                median(times),
                gib(used),
                gib(memory::vision_buffer_bytes(format, patches)),
            );
        }
        Ok(())
    }
}
