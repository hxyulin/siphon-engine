//! Device memory whose virtual addresses stay fixed while its physical pages come and go.
//!
//! A [`PagePool`] creates physical pages of the allocation granularity once, with
//! `cuMemCreate`. A [`Pooled`] buffer reserves virtual addresses for its largest size with
//! `cuMemAddressReserve`, which takes no memory, and maps pages from the pool at its start
//! as it grows ([`Pooled::map_to`]), and unmaps them when it shrinks ([`Pooled::unmap_from`]).
//! Kernels and CUDA graphs keep the buffer's address, since it never moves.
//!
//! A page may be mapped into several buffers at once ([`Pooled::map_shared`]): a page is an
//! `Arc`, and it returns to the pool when its last mapping goes. Whoever unmaps a page must
//! first make sure no queued work reads or writes it.

use std::sync::{Arc, Mutex, PoisonError};

use anyhow::{Context as _, ensure};
use cudarc::driver::{CudaContext, CudaStream, sys};

/// What `cuMemCreate` and `cuMemSetAccess` describe: device memory of one device.
fn properties(device: i32) -> sys::CUmemAllocationProp {
    // SAFETY: the struct is plain old data, for which all zeros is a valid value.
    let mut prop: sys::CUmemAllocationProp = unsafe { std::mem::zeroed() };
    prop.type_ = sys::CUmemAllocationType::CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location = sys::CUmemLocation {
        type_: sys::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE,
        id: device,
    };
    prop
}

/// The device's allocation granularities in bytes: the minimum and the recommended.
pub fn granularity(ctx: &Arc<CudaContext>) -> anyhow::Result<(usize, usize)> {
    ctx.bind_to_thread()?;
    let prop = properties(ctx.cu_device());
    let get = |flag| -> anyhow::Result<usize> {
        let mut bytes = 0usize;
        // SAFETY: `prop` describes the context's device, and `bytes` is what the call writes.
        unsafe { sys::cuMemGetAllocationGranularity(&raw mut bytes, &raw const prop, flag) }
            .result()
            .context("reading the allocation granularity")?;
        Ok(bytes)
    };
    Ok((
        get(sys::CUmemAllocationGranularity_flags::CU_MEM_ALLOC_GRANULARITY_MINIMUM)?,
        get(sys::CUmemAllocationGranularity_flags::CU_MEM_ALLOC_GRANULARITY_RECOMMENDED)?,
    ))
}

struct PoolShared {
    ctx: Arc<CudaContext>,
    page: usize,
    total: usize,
    free: Mutex<Vec<sys::CUmemGenericAllocationHandle>>,
}

impl PoolShared {
    fn free(&self) -> std::sync::MutexGuard<'_, Vec<sys::CUmemGenericAllocationHandle>> {
        self.free.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

impl Drop for PoolShared {
    fn drop(&mut self) {
        let _ = self.ctx.bind_to_thread();
        for &handle in self
            .free
            .get_mut()
            .unwrap_or_else(PoisonError::into_inner)
            .iter()
        {
            // SAFETY: every page is back in the free list, so none is mapped.
            let _ = unsafe { sys::cuMemRelease(handle) };
        }
    }
}

/// Physical device pages of the allocation granularity, created once.
#[derive(Clone)]
pub struct PagePool(Arc<PoolShared>);

impl std::fmt::Debug for PagePool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PagePool")
            .field("page", &self.0.page)
            .field("total", &self.0.total)
            .field("free", &self.free_pages())
            .finish()
    }
}

/// One physical page, mapped wherever its `Arc`s are. It returns to its pool when the last
/// one goes.
pub struct Page {
    handle: sys::CUmemGenericAllocationHandle,
    pool: Arc<PoolShared>,
}

impl Drop for Page {
    fn drop(&mut self) {
        self.pool.free().push(self.handle);
    }
}

impl PagePool {
    /// Creates `pages` pages of the device's minimum granularity.
    pub fn new(ctx: &Arc<CudaContext>, pages: usize) -> anyhow::Result<Self> {
        let (page, _) = granularity(ctx)?;
        Self::with_page(ctx, pages, page)
    }

    /// Creates `pages` pages of `page` bytes, a multiple of the granularity.
    pub fn with_page(ctx: &Arc<CudaContext>, pages: usize, page: usize) -> anyhow::Result<Self> {
        let (min, _) = granularity(ctx)?;
        ensure!(
            page > 0 && page.is_multiple_of(min),
            "a page of {page} bytes is not a multiple of the granularity, {min}"
        );
        let prop = properties(ctx.cu_device());
        let mut free = Vec::with_capacity(pages);
        let shared = |free| {
            Arc::new(PoolShared {
                ctx: Arc::clone(ctx),
                page,
                total: pages,
                free: Mutex::new(free),
            })
        };
        for created in 0..pages {
            let mut handle = 0;
            // SAFETY: `prop` describes the context's device, which is bound to the thread.
            let made =
                unsafe { sys::cuMemCreate(&raw mut handle, page, &raw const prop, 0) }.result();
            if let Err(e) = made {
                // The pages so far go back to the driver with the pool.
                drop(shared(free));
                return Err(e).with_context(|| {
                    format!(
                        "creating page {created} of {pages} ({} MiB each)",
                        page >> 20
                    )
                });
            }
            free.push(handle);
        }
        Ok(Self(shared(free)))
    }

    #[must_use]
    pub fn page_bytes(&self) -> usize {
        self.0.page
    }

    #[must_use]
    pub fn total_pages(&self) -> usize {
        self.0.total
    }

    #[must_use]
    pub fn free_pages(&self) -> usize {
        self.0.free().len()
    }

    fn take(&self) -> Option<Arc<Page>> {
        let handle = self.0.free().pop()?;
        Some(Arc::new(Page {
            handle,
            pool: Arc::clone(&self.0),
        }))
    }
}

/// Device memory at a fixed virtual address, of up to its reserved size, of which the first
/// [`Pooled::mapped_bytes`] are backed by pages of its pool.
pub struct Pooled {
    base: sys::CUdeviceptr,
    reserved: usize,
    pages: Vec<Arc<Page>>,
    pool: PagePool,
}

impl std::fmt::Debug for Pooled {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Pooled")
            .field("reserved", &self.reserved)
            .field("pages", &self.pages.len())
            .finish_non_exhaustive()
    }
}

impl Pooled {
    /// Reserves virtual addresses for `bytes`, rounded up to whole pages, with none mapped.
    pub fn reserve(pool: &PagePool, bytes: usize) -> anyhow::Result<Self> {
        let ctx = &pool.0.ctx;
        ctx.bind_to_thread()?;
        let reserved = bytes.max(1).next_multiple_of(pool.0.page);
        let mut base = 0;
        // SAFETY: the call writes the reserved range's start to `base`.
        unsafe { sys::cuMemAddressReserve(&raw mut base, reserved, pool.0.page, 0, 0) }
            .result()
            .with_context(|| format!("reserving {} MiB of addresses", reserved >> 20))?;
        Ok(Self {
            base,
            reserved,
            pages: Vec::new(),
            pool: pool.clone(),
        })
    }

    /// The buffer's device address.
    #[must_use]
    pub fn ptr(&self) -> u64 {
        self.base
    }

    #[must_use]
    pub fn reserved_bytes(&self) -> usize {
        self.reserved
    }

    #[must_use]
    pub fn mapped_bytes(&self) -> usize {
        self.pages.len() * self.pool.0.page
    }

    #[must_use]
    pub fn page_bytes(&self) -> usize {
        self.pool.0.page
    }

    /// The pool the pages come from.
    #[must_use]
    pub fn pool(&self) -> &PagePool {
        &self.pool
    }

    /// The mapped pages, from the start.
    #[must_use]
    pub fn pages(&self) -> &[Arc<Page>] {
        &self.pages
    }

    /// Pages more that [`Self::map_to`] `bytes` would take from the pool.
    #[must_use]
    pub fn pages_to(&self, bytes: usize) -> usize {
        bytes
            .min(self.reserved)
            .div_ceil(self.pool.0.page)
            .saturating_sub(self.pages.len())
    }

    /// Maps pages until at least the first `bytes` (at most the reserved size) are backed,
    /// without zeroing the newly mapped pages. Returns the number of newly mapped bytes.
    /// Fails, mapping nothing, if the pool lacks them.
    pub fn map_to_uninit(&mut self, bytes: usize) -> anyhow::Result<usize> {
        let more = self.pages_to(bytes);
        if more == 0 {
            return Ok(0);
        }
        let mut pages = Vec::with_capacity(more);
        for _ in 0..more {
            let Some(page) = self.pool.take() else {
                anyhow::bail!(
                    "the KV pool is out of pages: {more} more needed, {} free of {}",
                    pages.len(),
                    self.pool.0.total
                );
            };
            pages.push(page);
        }
        self.map_pages(pages)?;
        Ok(more * self.pool.0.page)
    }

    /// Maps pages until at least the first `bytes` (at most the reserved size) are backed,
    /// and zeroes the new pages on `stream`. Fails, mapping nothing, if the pool lacks them.
    pub fn map_to(&mut self, bytes: usize, stream: &CudaStream) -> anyhow::Result<()> {
        let start = self.mapped_bytes();
        let added = self.map_to_uninit(bytes)?;
        if added == 0 {
            return Ok(());
        }
        // SAFETY: the range was just mapped with read and write access, and work queued
        // on `stream` after this sees the zeros.
        unsafe { sys::cuMemsetD8Async(self.base + start as u64, 0, added, stream.cu_stream()) }
            .result()
            .context("zeroing new pages")?;
        Ok(())
    }

    /// Zeroes every currently mapped byte of this reservation in one async memset on
    /// `stream`. Does nothing when no pages are mapped.
    pub fn zero_mapped(&self, stream: &CudaStream) -> anyhow::Result<()> {
        let mapped = self.mapped_bytes();
        if mapped == 0 {
            return Ok(());
        }
        // SAFETY: `[self.base, self.base + mapped)` is currently mapped with RW access.
        unsafe { sys::cuMemsetD8Async(self.base, 0, mapped, stream.cu_stream()) }
            .result()
            .context("zeroing mapped pages")?;
        Ok(())
    }

    /// Maps `pages`, which other buffers may map too, after the mapped ones, without
    /// zeroing them.
    pub fn map_shared(&mut self, pages: &[Arc<Page>]) -> anyhow::Result<()> {
        self.map_pages(pages.to_vec())
    }

    fn map_pages(&mut self, pages: Vec<Arc<Page>>) -> anyhow::Result<()> {
        let page = self.pool.0.page;
        let start = self.mapped_bytes();
        ensure!(
            start + pages.len() * page <= self.reserved,
            "{} pages past {start} bytes overflow a reservation of {}",
            pages.len(),
            self.reserved
        );
        if pages.is_empty() {
            return Ok(());
        }
        self.pool.0.ctx.bind_to_thread()?;
        for (i, p) in pages.iter().enumerate() {
            let at = self.base + (start + i * page) as u64;
            // SAFETY: `at` is a page-aligned unmapped page of this reservation, and the
            // handle a live page of the pool's size.
            unsafe { sys::cuMemMap(at, page, 0, p.handle, 0) }
                .result()
                .context("mapping a page")?;
        }
        let access = sys::CUmemAccessDesc {
            location: properties(self.pool.0.ctx.cu_device()).location,
            flags: sys::CUmemAccess_flags::CU_MEM_ACCESS_FLAGS_PROT_READWRITE,
        };
        // SAFETY: the range was just mapped.
        unsafe {
            sys::cuMemSetAccess(
                self.base + start as u64,
                pages.len() * page,
                &raw const access,
                1,
            )
        }
        .result()
        .context("granting access to new pages")?;
        self.pages.extend(pages);
        Ok(())
    }

    /// Unmaps every page past the first `bytes`, rounded up to a page, in one contiguous
    /// `cuMemUnmap` call. Queued work must not touch them any more.
    pub fn unmap_from(&mut self, bytes: usize) -> anyhow::Result<()> {
        let keep = bytes.div_ceil(self.pool.0.page).min(self.pages.len());
        if keep == self.pages.len() {
            return Ok(());
        }
        self.pool.0.ctx.bind_to_thread()?;
        let page = self.pool.0.page;
        let count = self.pages.len() - keep;
        // SAFETY: `[base + keep * page, base + len * page)` is contiguously mapped, and
        // the caller promises no queued work uses it.
        unsafe { sys::cuMemUnmap(self.base + (keep * page) as u64, count * page) }
            .result()
            .context("unmapping pages")?;
        self.pages.truncate(keep);
        Ok(())
    }
}

impl Drop for Pooled {
    fn drop(&mut self) {
        let _ = self.unmap_from(0);
        let _ = self.pool.0.ctx.bind_to_thread();
        // SAFETY: nothing is mapped in the range any more.
        let _ = unsafe { sys::cuMemAddressFree(self.base, self.reserved) };
    }
}
