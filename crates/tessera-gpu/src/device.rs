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
pub struct DeviceBuffer<T>(CudaSlice<T>);

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
    stream: Arc<CudaStream>,
    vector_add: CudaFunction,
}

impl Device {
    /// Opens device `ordinal` and loads the kernels.
    pub fn new(ordinal: usize) -> Result<Self, GpuError> {
        if cfg!(tessera_gpu_kernels_skipped) || VECTOR_ADD_FATBIN.is_empty() {
            return Err(GpuError::KernelsNotBuilt);
        }
        let context = CudaContext::new(ordinal)?;
        let module = context.load_module(Ptx::from_binary(VECTOR_ADD_FATBIN.to_vec()))?;
        let vector_add = module.load_function("vector_add")?;
        Ok(Self {
            stream: context.default_stream(),
            vector_add,
        })
    }

    /// Copies `host` into a new device buffer.
    pub fn upload<T: DeviceRepr>(&self, host: &[T]) -> Result<DeviceBuffer<T>, GpuError> {
        Ok(DeviceBuffer(self.stream.clone_htod(host)?))
    }

    /// A new device buffer of `len` zeroed elements.
    pub fn zeroed<T: DeviceRepr + ValidAsZeroBits>(
        &self,
        len: usize,
    ) -> Result<DeviceBuffer<T>, GpuError> {
        Ok(DeviceBuffer(self.stream.alloc_zeros(len)?))
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
