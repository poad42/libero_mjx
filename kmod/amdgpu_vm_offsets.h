/* SPDX-License-Identifier: GPL-2.0 */
/*
 * amdgpu_vm_offsets.h - byte offsets of the struct amdgpu_vm fields we touch,
 * for the gfx1201 mempool TLB-race trace / fix modules.
 *
 * These modules deliberately do NOT #include amdgpu's private headers
 * (drivers/gpu/drm/amd/amdgpu/amdgpu_vm.h pulls in the entire driver).
 * Instead we read/write a handful of fields by offset. The offsets below were
 * extracted with pahole from the *exact* amdgpu.ko of the target build:
 *
 *   kernel : 6.19.0-amd-staging-p2p-amd-staging-p2p  (gcc 16.1.1)
 *   amdgpu : srcversion E0019EF19146799273262E3
 *   tree   : ~/hip_p2p_pcie/vendor/linux-amd-staging-drm-next
 *
 *   $ pahole -C amdgpu_vm amdgpu.ko
 *     atomic64_t tlb_seq;               // 1112
 *     struct dma_fence *last_tlb_flush; // 1120
 *     atomic64_t kfd_last_flushed_seq;  // 1128
 *     unsigned int pasid;               // 1160
 *     bool is_compute_context;          // 2968
 *     bool need_tlb_fence;              // 2969
 *
 * The module init guards on amdgpu's srcversion (see AMDGPU_SRCVERSION_EXPECT)
 * and refuses to load if it does not match, so a layout change cannot lead to
 * touching the wrong field.
 *
 * Re-verified against the LIVE amdgpu.ko (pahole v1.31, BTF) on the running
 * 6.19.0-amd-staging-p2p kernel: all six offsets below + the srcversion match
 * exactly (struct amdgpu_vm size 2992). `tlbfix.sh check` re-runs this check.
 */
#ifndef AMDGPU_VM_OFFSETS_H
#define AMDGPU_VM_OFFSETS_H

#define AMDGPU_SRCVERSION_EXPECT "8F7D3C7777925E8D9DCD377"

#define AVM_OFF_TLB_SEQ              1112  /* atomic64_t */
#define AVM_OFF_LAST_TLB_FLUSH       1120  /* struct dma_fence * */
#define AVM_OFF_KFD_LAST_FLUSHED_SEQ 1128  /* atomic64_t */
#define AVM_OFF_PASID                1160  /* unsigned int */
#define AVM_OFF_IS_COMPUTE_CONTEXT   2968  /* bool */
#define AVM_OFF_NEED_TLB_FENCE       2969  /* bool */

static inline long long avm_tlb_seq(const void *vm)
{
	return (long long)atomic64_read((const atomic64_t *)
				((const char *)vm + AVM_OFF_TLB_SEQ));
}
static inline long long avm_kfd_last_flushed_seq(const void *vm)
{
	return (long long)atomic64_read((const atomic64_t *)
				((const char *)vm + AVM_OFF_KFD_LAST_FLUSHED_SEQ));
}
static inline unsigned int avm_pasid(const void *vm)
{
	return *(const unsigned int *)((const char *)vm + AVM_OFF_PASID);
}
static inline bool avm_is_compute(const void *vm)
{
	return *(const bool *)((const char *)vm + AVM_OFF_IS_COMPUTE_CONTEXT);
}
/* vm->last_tlb_flush: the dma_fence that signals when the most recent fenced
 * GPU TLB flush for this VM has actually completed on hardware. */
static inline void *avm_last_tlb_flush(const void *vm)
{
	return *(void * const *)((const char *)vm + AVM_OFF_LAST_TLB_FLUSH);
}
static inline bool avm_need_tlb_fence(const void *vm)
{
	return *(const bool *)((const char *)vm + AVM_OFF_NEED_TLB_FENCE);
}
static inline void avm_set_need_tlb_fence(void *vm, bool val)
{
	*(bool *)((char *)vm + AVM_OFF_NEED_TLB_FENCE) = val;
}

#endif /* AMDGPU_VM_OFFSETS_H */
