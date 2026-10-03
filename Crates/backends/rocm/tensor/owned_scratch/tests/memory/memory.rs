//! Test-only admission recorder: all operations delegate to the real provider.
#[rustfmt::skip]
use fusion_pcu::{
    PcuMemoryAllocationRequest,
    PcuMemoryPoolId,
    PcuMemoryPoolSnapshot,
    PcuMemoryProvider,
    PcuMemoryProviderError,
    PcuMemoryRange,
};
pub struct Memory<P> {
    pub provider: P,
    pub requests: Vec<PcuMemoryAllocationRequest>,
}
impl<P: PcuMemoryProvider> PcuMemoryProvider for Memory<P> {
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
        let resource = self.provider.allocate(request)?;
        self.requests.push(request);
        Ok(resource)
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
        self.provider.transfer_to(resource, offset_bytes, bytes)
    }
    fn transfer_from(
        &mut self,
        resource: &Self::Resource,
        offset_bytes: u64,
        bytes: &mut [u8],
    ) -> Result<(), PcuMemoryProviderError> {
        self.provider.transfer_from(resource, offset_bytes, bytes)
    }
}
