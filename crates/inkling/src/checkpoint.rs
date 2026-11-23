use std::{
    collections::BTreeMap,
    fs::File,
    ops::Range,
    path::{Path, PathBuf},
};

use anyhow::{Context as _, bail};
use memmap2::Mmap;
use safetensors::{Dtype, SafeTensors};

/// Every tensor in a directory of `.safetensors` files, memory-mapped.
///
/// All files in the directory are read, including ones the index omits,
/// such as the draft model's `mtp.safetensors`.
#[derive(Debug)]
pub struct Checkpoint {
    files: Vec<(PathBuf, Mmap)>,
    tensors: BTreeMap<String, TensorEntry>,
}

/// Location and layout of one stored tensor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorEntry {
    pub dtype: Dtype,
    pub shape: Vec<usize>,
    file: usize,
    bytes: Range<usize>,
}

impl TensorEntry {
    /// Stored size in bytes.
    #[must_use]
    pub fn byte_len(&self) -> usize {
        self.bytes.len()
    }
}

impl Checkpoint {
    /// Maps every `*.safetensors` file in `dir` and indexes its tensors.
    pub fn open(dir: impl AsRef<Path>) -> anyhow::Result<Self> {
        let dir = dir.as_ref();
        let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)
            .with_context(|| format!("listing {}", dir.display()))?
            .map(|entry| entry.map(|e| e.path()))
            .collect::<Result<_, _>>()?;
        paths.retain(|p| p.extension().is_some_and(|ext| ext == "safetensors"));
        paths.sort();
        if paths.is_empty() {
            bail!("no .safetensors files in {}", dir.display());
        }

        let mut files = Vec::with_capacity(paths.len());
        let mut tensors = BTreeMap::new();
        for (file_index, path) in paths.into_iter().enumerate() {
            let file = File::open(&path).with_context(|| format!("opening {}", path.display()))?;
            // SAFETY: checkpoint files are treated as read-only for the life of the process.
            // Another process truncating one while it is mapped would be a bug outside our control.
            let map = unsafe { Mmap::map(&file) }
                .with_context(|| format!("mapping {}", path.display()))?;
            let (header_len, metadata) = SafeTensors::read_metadata(&map)
                .with_context(|| format!("reading header of {}", path.display()))?;
            let data_start = 8 + header_len;
            for (name, info) in metadata.tensors() {
                let (start, end) = info.data_offsets;
                let entry = TensorEntry {
                    dtype: info.dtype,
                    shape: info.shape.clone(),
                    file: file_index,
                    bytes: data_start + start..data_start + end,
                };
                if tensors.insert(name.clone(), entry).is_some() {
                    bail!("tensor {name} appears in more than one file");
                }
            }
            files.push((path, map));
        }
        Ok(Self { files, tensors })
    }

    /// All tensors, sorted by name.
    pub fn tensors(&self) -> impl Iterator<Item = (&str, &TensorEntry)> {
        self.tensors
            .iter()
            .map(|(name, entry)| (name.as_str(), entry))
    }

    #[must_use]
    pub fn get(&self, name: &str) -> Option<&TensorEntry> {
        self.tensors.get(name)
    }

    /// Raw stored bytes of `entry`, borrowed from the mapping.
    #[must_use]
    pub fn bytes(&self, entry: &TensorEntry) -> &[u8] {
        &self.files[entry.file].1[entry.bytes.clone()]
    }

    /// Path of the file that stores `entry`.
    #[must_use]
    pub fn file_path(&self, entry: &TensorEntry) -> &Path {
        &self.files[entry.file].0
    }
}
