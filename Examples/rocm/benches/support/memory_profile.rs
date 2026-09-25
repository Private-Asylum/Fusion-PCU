//! Measure provider operations without changing their resource or transfer behavior.

use std::time::{
    Duration,
    Instant,
};

use fusion_pcu::{
    PcuMemoryAllocationRequest,
    PcuMemoryPoolId,
    PcuMemoryPoolSnapshot,
    PcuMemoryProvider,
    PcuMemoryProviderError,
    PcuMemoryRange,
};

#[derive(Clone, Copy, Debug, Default)]
pub struct MemoryProfile {
    pub allocations: usize,
    pub allocated_bytes: u64,
    pub allocation_time: Duration,
    pub uploads: usize,
    pub uploaded_bytes: usize,
    pub upload_time: Duration,
    pub downloads: usize,
    pub downloaded_bytes: usize,
    pub download_time: Duration,
}

pub struct ProfiledMemory<P> {
    provider: P,
    profile: MemoryProfile,
}

impl<P> ProfiledMemory<P> {
    pub const fn new(provider: P) -> Self {
        Self {
            provider,
            profile: MemoryProfile {
                allocations: 0,
                allocated_bytes: 0,
                allocation_time: Duration::ZERO,
                uploads: 0,
                uploaded_bytes: 0,
                upload_time: Duration::ZERO,
                downloads: 0,
                downloaded_bytes: 0,
                download_time: Duration::ZERO,
            },
        }
    }

    pub const fn profile(&self) -> MemoryProfile {
        self.profile
    }
}

impl<P: PcuMemoryProvider> PcuMemoryProvider for ProfiledMemory<P> {
    type Resource = P::Resource;
    type ImportDescriptor = P::ImportDescriptor;
    type Mapping<'a>
        = P::Mapping<'a>
    where
        Self: 'a;

    fn snapshot(
        &self,
        pool: PcuMemoryPoolId,
    ) -> Result<PcuMemoryPoolSnapshot, PcuMemoryProviderError> {
        self.provider.snapshot(pool)
    }

    fn allocate(
        &mut self,
        request: PcuMemoryAllocationRequest,
    ) -> Result<Self::Resource, PcuMemoryProviderError> {
        let start = Instant::now();
        let result = self.provider.allocate(request);
        self.profile.allocation_time += start.elapsed();
        self.profile.allocations += 1;
        self.profile.allocated_bytes += request.size_bytes;
        result
    }

    fn import(
        &mut self,
        descriptor: Self::ImportDescriptor,
    ) -> Result<Self::Resource, PcuMemoryProviderError> {
        self.provider.import(descriptor)
    }

    fn map<'a>(
        &'a mut self,
        resource: &'a mut Self::Resource,
        range: PcuMemoryRange,
    ) -> Result<Self::Mapping<'a>, PcuMemoryProviderError> {
        self.provider.map(resource, range)
    }

    fn transfer_to(
        &mut self,
        resource: &mut Self::Resource,
        offset_bytes: u64,
        bytes: &[u8],
    ) -> Result<(), PcuMemoryProviderError> {
        let start = Instant::now();
        let result = self.provider.transfer_to(resource, offset_bytes, bytes);
        self.profile.upload_time += start.elapsed();
        self.profile.uploads += 1;
        self.profile.uploaded_bytes += bytes.len();
        result
    }

    fn transfer_from(
        &mut self,
        resource: &Self::Resource,
        offset_bytes: u64,
        bytes: &mut [u8],
    ) -> Result<(), PcuMemoryProviderError> {
        let start = Instant::now();
        let result = self.provider.transfer_from(resource, offset_bytes, bytes);
        self.profile.download_time += start.elapsed();
        self.profile.downloads += 1;
        self.profile.downloaded_bytes += bytes.len();
        result
    }
}
