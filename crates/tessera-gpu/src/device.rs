// SPDX-License-Identifier: Apache-2.0 OR MIT

//! A CUDA device with this crate's kernels loaded, and the device buffers they operate on.
//!
//! The vector-add kernel is the template for later kernels (RFC-0018 'Kernels'): each kernel
//! family is compiled from `kernels/<family>.cu` into a fatbin, embedded in the binary, loaded
//! into the device once, and launched through a safe wrapper that checks lengths first.

use std::fmt;
use std::sync::Arc;

use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, DeviceRepr, DriverError, LaunchConfig,
    PushKernelArg, ValidAsZeroBits,
};
use cudarc::nvrtc::Ptx;

/// The fatbin of `kernels/vector_add.cu`, for `sm_90` and `sm_100` (empty when the build skipped
/// kernel compilation).
const VECTOR_ADD_FATBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/vector_add.fatbin"));
/// The fatbin of `kernels/commit.cu`: encoding and column hashing (#55).
const COMMIT_FATBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/commit.fatbin"));
/// The fatbin of `kernels/trees.cu`: product- and fraction-tree levels and their sumchecks (#63,
/// #64).
const TREES_FATBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/trees.fatbin"));

/// An error from the GPU backend. It carries public data only (05 §3).
#[derive(Debug)]
pub enum GpuError {
    /// The CUDA driver reported an error (`ERR-RES-DEVICE` in 04 §3).
    Driver(DriverError),
    /// This build skipped kernel compilation (`TESSERA_GPU_SKIP_KERNELS=1`) and cannot run any.
    KernelsNotBuilt,
    /// Two operands have different lengths.
    LengthMismatch {
        /// Length of the left operand.
        left: usize,
        /// Length of the right operand.
        right: usize,
    },
    /// A buffer does not hold the number of elements the operation needs.
    BufferLength {
        /// Which operand.
        operand: &'static str,
        /// The length the operation needs.
        expected: usize,
        /// The buffer's length.
        found: usize,
    },
    /// A shape parameter is outside what the kernels support.
    Unsupported(&'static str),
}

impl fmt::Display for GpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Driver(e) => write!(f, "CUDA driver error: {e}"),
            Self::KernelsNotBuilt => f.write_str(
                "this build has no CUDA kernels (built with TESSERA_GPU_SKIP_KERNELS=1)",
            ),
            Self::LengthMismatch { left, right } => {
                write!(f, "operand lengths differ: {left} and {right}")
            }
            Self::BufferLength {
                operand,
                expected,
                found,
            } => write!(f, "{operand} holds {found} elements, not {expected}"),
            Self::Unsupported(what) => write!(f, "unsupported shape: {what}"),
        }
    }
}

impl std::error::Error for GpuError {}

impl From<DriverError> for GpuError {
    fn from(e: DriverError) -> Self {
        Self::Driver(e)
    }
}

/// A buffer of `T` in device memory, owned by one [`Device`].
pub struct DeviceBuffer<T>(pub(crate) CudaSlice<T>);

impl<T> DeviceBuffer<T> {
    /// The number of elements.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Whether the buffer has no elements.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

/// One CUDA device with this crate's kernels loaded, and the stream all its work is ordered on.
pub struct Device {
    pub(crate) stream: Arc<CudaStream>,
    vector_add: CudaFunction,
    pub(crate) commit: crate::commit::Kernels,
    pub(crate) trees: crate::trees::Kernels,
    pub(crate) fraction: crate::fraction::Kernels,
    pub(crate) level: crate::level::Kernels,
}

impl Device {
    /// Opens device `ordinal` and loads the kernels.
    pub fn new(ordinal: usize) -> Result<Self, GpuError> {
        if cfg!(tessera_gpu_kernels_skipped)
            || VECTOR_ADD_FATBIN.is_empty()
            || COMMIT_FATBIN.is_empty()
            || TREES_FATBIN.is_empty()
        {
            return Err(GpuError::KernelsNotBuilt);
        }
        let context = CudaContext::new(ordinal)?;
        let module = context.load_module(Ptx::from_binary(VECTOR_ADD_FATBIN.to_vec()))?;
        let vector_add = module.load_function("vector_add")?;
        let commit = context.load_module(Ptx::from_binary(COMMIT_FATBIN.to_vec()))?;
        let trees = context.load_module(Ptx::from_binary(TREES_FATBIN.to_vec()))?;
        Ok(Self {
            stream: context.default_stream(),
            vector_add,
            commit: crate::commit::Kernels::load(&commit)?,
            trees: crate::trees::Kernels::load(&trees)?,
            fraction: crate::fraction::Kernels::load(&trees)?,
            level: crate::level::Kernels::load(&trees)?,
        })
    }

    /// Copies `host` into a new device buffer.
    pub fn upload<T: DeviceRepr>(&self, host: &[T]) -> Result<DeviceBuffer<T>, GpuError> {
        Ok(DeviceBuffer(self.stream.clone_htod(host)?))
    }

    /// Copies `host` into the first `host.len()` elements of `buffer`.
    pub fn upload_into<T: DeviceRepr>(
        &self,
        host: &[T],
        buffer: &mut DeviceBuffer<T>,
    ) -> Result<(), GpuError> {
        if host.len() > buffer.len() {
            return Err(GpuError::LengthMismatch {
                left: host.len(),
                right: buffer.len(),
            });
        }
        let mut view = buffer.0.slice_mut(0..host.len());
        self.stream.memcpy_htod(host, &mut view)?;
        Ok(())
    }

    /// Copies field elements into the first `host.len()` words of `buffer`, as the canonical
    /// `u64` each holds, without a host copy.
    pub fn upload_field_into(
        &self,
        host: &[tessera_field::Goldilocks],
        buffer: &mut DeviceBuffer<u64>,
    ) -> Result<(), GpuError> {
        #[allow(unsafe_code)]
        // SAFETY: `Goldilocks` is `#[repr(transparent)]` over its canonical `u64`, so the slice's
        // memory is `host.len()` initialized, aligned `u64`s, borrowed for the copy's duration.
        let words = unsafe { std::slice::from_raw_parts(host.as_ptr().cast::<u64>(), host.len()) };
        self.upload_into(words, buffer)
    }

    /// Overwrites every element of `buffer` with zero: device memory that held secret-class
    /// values is cleared before it is released or reused for public data (05 §3 OBS-6).
    pub fn clear<T: DeviceRepr + ValidAsZeroBits>(
        &self,
        buffer: &mut DeviceBuffer<T>,
    ) -> Result<(), GpuError> {
        self.stream.memset_zeros(&mut buffer.0)?;
        Ok(())
    }

    /// A new device buffer of `len` zeroed elements.
    pub fn zeroed<T: DeviceRepr + ValidAsZeroBits>(
        &self,
        len: usize,
    ) -> Result<DeviceBuffer<T>, GpuError> {
        Ok(DeviceBuffer(self.stream.alloc_zeros(len)?))
    }

    /// Waits until all work queued on the device completes.
    pub fn synchronize(&self) -> Result<(), GpuError> {
        Ok(self.stream.synchronize()?)
    }

    /// Copies a device buffer back to the host, after the work queued before it completes.
    pub fn download<T: DeviceRepr>(&self, buffer: &DeviceBuffer<T>) -> Result<Vec<T>, GpuError> {
        Ok(self.stream.clone_dtoh(&buffer.0)?)
    }

    /// `a[i] + b[i]` modulo `2^64`, element by element.
    pub fn vector_add(
        &self,
        a: &DeviceBuffer<u64>,
        b: &DeviceBuffer<u64>,
    ) -> Result<DeviceBuffer<u64>, GpuError> {
        if a.len() != b.len() {
            return Err(GpuError::LengthMismatch {
                left: a.len(),
                right: b.len(),
            });
        }
        let n = a.len();
        let mut out = self.zeroed::<u64>(n)?;
        if n == 0 {
            return Ok(out);
        }
        // `for_num_elems` takes a u32; the kernel's grid-stride loop covers any n with that grid.
        let config = LaunchConfig::for_num_elems(u32::try_from(n).unwrap_or(u32::MAX));
        let n = n as u64;
        let mut launch = self.stream.launch_builder(&self.vector_add);
        launch.arg(&mut out.0).arg(&a.0).arg(&b.0).arg(&n);
        #[allow(unsafe_code)]
        // SAFETY: the arguments match `vector_add(uint64_t *out, const uint64_t *a,
        // const uint64_t *b, uint64_t n)` in kernels/vector_add.cu in number, order and type, and
        // all three buffers hold exactly `n` u64 values, so every index the kernel touches
        // (`i < n`) is in bounds. The buffers outlive the launch: the stream orders the launch
        // before any later use of them, including the download.
        unsafe { launch.launch(config) }?;
        Ok(out)
    }
}
