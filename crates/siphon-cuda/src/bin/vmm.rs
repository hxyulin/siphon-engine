//! Measures the virtual memory calls that pooled KV caches use: the allocation
//! granularity, creating pages, mapping and unmapping them, and copy bandwidth through
//! mapped pages against `cuMemAllocAsync` memory.
//!
//! `vmm [--pages 2048] [--caches 16]`: maps chunks of one page in each of `--caches`
//! buffers, as a 2,048-position chunk of FP8 global caches does.

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("vmm needs Linux with an NVIDIA GPU");
}

#[cfg(target_os = "linux")]
#[expect(clippy::too_many_lines, reason = "one measurement after another")]
fn main() -> anyhow::Result<()> {
    use std::time::Instant;

    use siphon_cuda::{
        Gpu,
        vmm::{PagePool, Pooled, granularity},
    };

    let mut pages = 2048;
    let mut caches = 16;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        let mut value = || anyhow::Context::context(args.next(), "missing value");
        match arg.as_str() {
            "--pages" => pages = value()?.parse()?,
            "--caches" => caches = value()?.parse()?,
            _ => anyhow::bail!("unknown argument {arg}"),
        }
    }
    let gpu = Gpu::new(0)?;
    let ctx = gpu.context();
    let stream = gpu.stream();
    let (min, rec) = granularity(ctx)?;
    println!(
        "granularity: minimum {} KiB, recommended {} KiB",
        min >> 10,
        rec >> 10
    );

    let t = Instant::now();
    let pool = PagePool::new(ctx, pages)?;
    let create = t.elapsed().as_secs_f64();
    let page = pool.page_bytes();
    println!(
        "created {pages} pages of {} MiB in {:.1} ms ({:.1} us each)",
        page >> 20,
        create * 1e3,
        create * 1e6 / pages as f64
    );

    // Buffers as big as the pool each: a chunk maps one page in every buffer.
    let mut buffers: Vec<Pooled> = (0..caches)
        .map(|_| Pooled::reserve(&pool, pages * page))
        .collect::<anyhow::Result<_>>()?;
    let chunks = pages / caches;
    let mut map_times = Vec::with_capacity(chunks);
    for c in 0..chunks {
        let t = Instant::now();
        for b in &mut buffers {
            b.map_to((c + 1) * page, stream)?;
        }
        map_times.push(t.elapsed().as_secs_f64());
    }
    stream.synchronize()?;
    let mut unmap_times = Vec::with_capacity(chunks);
    for c in (0..chunks).rev() {
        let t = Instant::now();
        for b in &mut buffers {
            b.unmap_from(c * page)?;
        }
        unmap_times.push(t.elapsed().as_secs_f64());
    }
    let summary = |name: &str, times: &mut Vec<f64>| {
        times.sort_by(f64::total_cmp);
        let mean = times.iter().sum::<f64>() / times.len() as f64;
        println!(
            "{name} a chunk of {caches} pages: median {:.1} us, mean {:.1} us, max {:.1} us \
             ({:.2} us per page)",
            times[times.len() / 2] * 1e6,
            mean * 1e6,
            times[times.len() - 1] * 1e6,
            mean * 1e6 / caches as f64
        );
    };
    summary("map+access+zero", &mut map_times);
    summary("unmap", &mut unmap_times);

    // A restore maps many chunks at once: every chunk of each buffer in one call.
    let t = Instant::now();
    for b in &mut buffers {
        b.map_to(chunks * page, stream)?;
    }
    let bulk = t.elapsed().as_secs_f64();
    stream.synchronize()?;
    let t = Instant::now();
    for b in &mut buffers {
        b.unmap_from(0)?;
    }
    let bulk_unmap = t.elapsed().as_secs_f64();
    println!(
        "{chunks} chunks at once: map+access+zero {:.1} ms ({:.1} us per page), unmap {:.1} ms",
        bulk * 1e3,
        bulk * 1e6 / (chunks * caches) as f64,
        bulk_unmap * 1e3
    );

    // Whether unmapping waits for unrelated queued work: queue a long copy, then unmap.
    let big = pages / 4 * page;
    let src = stream.alloc_zeros::<u8>(big)?;
    let mut dst = stream.alloc_zeros::<u8>(big)?;
    for b in &mut buffers {
        b.map_to(page, stream)?;
    }
    stream.synchronize()?;
    let t = Instant::now();
    for _ in 0..8 {
        stream.memcpy_dtod(&src, &mut dst)?;
    }
    let queued = t.elapsed().as_secs_f64();
    let t = Instant::now();
    for b in &mut buffers {
        b.unmap_from(0)?;
    }
    let unmap_busy = t.elapsed().as_secs_f64();
    let t = Instant::now();
    stream.synchronize()?;
    let rest = t.elapsed().as_secs_f64();
    println!(
        "with 8 copies of {} MiB queued ({:.2} ms to queue): unmapping {caches} pages took \
         {:.1} us, the copies finished {:.1} ms later",
        big >> 20,
        queued * 1e3,
        unmap_busy * 1e6,
        rest * 1e3
    );
    let t = Instant::now();
    for b in &mut buffers {
        b.map_to(page, stream)?;
    }
    let map_busy = t.elapsed().as_secs_f64();
    stream.synchronize()?;
    println!("mapping {caches} pages took {:.1} us", map_busy * 1e6);
    drop(buffers);
    drop(src);
    drop(dst);

    // Copy bandwidth: half the pool as one mapped buffer against an allocated one.
    let half = pages / 2 * page;
    let mut mapped = Pooled::reserve(&pool, half)?;
    mapped.map_to(half, stream)?;
    let plain = stream.alloc_zeros::<u8>(half)?;
    stream.synchronize()?;
    let copy = |from: u64, to: u64| -> anyhow::Result<f64> {
        let mut best = f64::MAX;
        for _ in 0..5 {
            let t = Instant::now();
            // SAFETY: both ranges are live device memory of `half` bytes.
            unsafe {
                cudarc::driver::sys::cuMemcpyDtoDAsync_v2(to, from, half, stream.cu_stream())
            }
            .result()?;
            stream.synchronize()?;
            best = best.min(t.elapsed().as_secs_f64());
        }
        Ok(2.0 * half as f64 / best / 1e9)
    };
    let plain_ptr = {
        use cudarc::driver::DevicePtr as _;
        plain.device_ptr(stream).0
    };
    println!(
        "copy of {} MiB: allocated to mapped {:.0} GB/s, mapped to allocated {:.0} GB/s",
        half >> 20,
        copy(plain_ptr, mapped.ptr())?,
        copy(mapped.ptr(), plain_ptr)?
    );
    Ok(())
}
