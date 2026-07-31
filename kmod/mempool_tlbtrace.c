// SPDX-License-Identifier: GPL-2.0
/*
 * mempool_tlbtrace.c - PHASE 1, trace-only kprobe instrumentation for the
 * gfx1201 hipMallocAsync stream-ordered-pool TLB-flush race.
 *
 * Goal: with NO behaviour change, confirm *which* amdgpu page-table / TLB-flush
 * path the stream-ordered pool's trim -> re-map cycle takes, and prove the
 * missing-flush / async-tlb_seq ordering window described in MEMPOOL_REPORT.md
 * and ASM_BYPASS_REPORT.md (corruption = stale VA->PA translation after re-map).
 *
 * It installs kprobes on the candidate functions and logs, in order, every time
 * the target process (matched by a substring of current->comm, default
 * "mempool") drives one of them, capturing the decisive fields:
 *
 *   amdgpu_vm_update_range       vm, flush_tlb arg, is_compute_context,
 *                                need_tlb_fence, tlb_seq (the PT update + the
 *                                request to flush)
 *   amdgpu_vm_tlb_fence_create   the *fenced* (correct) flush path -- if this
 *                                never fires for the pool's VM, the flush is the
 *                                racy async tlb_seq increment only
 *   amdgpu_gmc_flush_gpu_tlb_pasid  the real HW TLB invalidate (pasid, type)
 *   amdgpu_vm_flush_compute_tlb  the racy "xchg seq, skip if equal" KFD check
 *   amdgpu_amdkfd_gpuvm_(un)map_memory_to_gpu  KFD map/unmap entry (KFD path)
 *   amdgpu_gem_va_ioctl          graphics render-node VA map/unmap (gfx path)
 *
 * Safety: every handler does a cheap current->comm substring test first and
 * returns immediately for non-target processes, so it does not perturb the
 * other GPU workloads sharing this node. Logging is capped (maxlog).
 *
 * x86-64 only (uses the SysV arg registers).
 */
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/kprobes.h>
#include <linux/sched.h>
#include <linux/string.h>
#include <linux/atomic.h>
#include <linux/ktime.h>
#include <linux/fs.h>
#include <linux/uaccess.h>
#include <asm/ptrace.h>

#include "amdgpu_vm_offsets.h"

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Agent-Kmod");
MODULE_DESCRIPTION("Trace-only kprobes for the gfx1201 mempool TLB-flush race");
MODULE_VERSION("1.0");

static char *targetcomm = "mempool";
module_param(targetcomm, charp, 0644);
MODULE_PARM_DESC(targetcomm, "substring of current->comm to trace (default mempool)");

static int maxlog = 600;
module_param(maxlog, int, 0644);
MODULE_PARM_DESC(maxlog, "max number of event lines to emit (default 600)");

static int strict_srcversion = 1;
module_param(strict_srcversion, int, 0644);
MODULE_PARM_DESC(strict_srcversion, "refuse to load if amdgpu srcversion mismatches (default 1)");

static atomic_t logged = ATOMIC_INIT(0);
static atomic64_t seqno = ATOMIC64_INIT(0);

static bool is_target(void)
{
	return strstr(current->comm, targetcomm) != NULL;
}

static bool budget(void)
{
	return atomic_inc_return(&logged) <= maxlog;
}

#define EV(fmt, ...) do {                                                   \
	if (is_target() && budget())                                       \
		pr_info("tlbtrace %6lld [%s/%d] " fmt "\n",                \
			(long long)atomic64_inc_return(&seqno),            \
			current->comm, current->pid, ##__VA_ARGS__);       \
} while (0)

/* ---- amdgpu_vm_update_range(adev, vm, immediate, unlocked, flush_tlb, ...) */
static int pre_update_range(struct kprobe *p, struct pt_regs *regs)
{
	void *vm = (void *)regs->si;
	bool flush_tlb = (regs->r8 & 0xff) != 0;

	if (is_target() && vm && budget())
		pr_info("tlbtrace %6lld [%s/%d] update_range vm=%px flush_tlb=%d compute=%d need_tlb_fence=%d tlb_seq=%lld\n",
			(long long)atomic64_inc_return(&seqno),
			current->comm, current->pid, vm, flush_tlb,
			avm_is_compute(vm), avm_need_tlb_fence(vm),
			avm_tlb_seq(vm));
	return 0;
}

/* ---- amdgpu_vm_tlb_fence_create(adev, vm, fence) : the *correct* fenced flush */
static int pre_tlb_fence_create(struct kprobe *p, struct pt_regs *regs)
{
	void *vm = (void *)regs->si;

	if (is_target() && vm && budget())
		pr_info("tlbtrace %6lld [%s/%d] *** tlb_fence_create vm=%px pasid=%u tlb_seq=%lld  (FENCED flush path)\n",
			(long long)atomic64_inc_return(&seqno),
			current->comm, current->pid, vm,
			avm_pasid(vm), avm_tlb_seq(vm));
	return 0;
}

/* ---- amdgpu_gmc_flush_gpu_tlb_pasid(adev, pasid, flush_type, all_hub, inst) */
static int pre_flush_pasid(struct kprobe *p, struct pt_regs *regs)
{
	unsigned int pasid = (unsigned int)regs->si;
	unsigned int ftype = (unsigned int)regs->dx;

	EV("=== HW TLB FLUSH pasid=%u type=%u (gmc_flush_gpu_tlb_pasid)",
	   pasid, ftype);
	return 0;
}

/* ---- amdgpu_vm_flush_compute_tlb(adev, vm, flush_type, xcc_mask) */
static int pre_compute_tlb(struct kprobe *p, struct pt_regs *regs)
{
	void *vm = (void *)regs->si;
	unsigned int ftype = (unsigned int)regs->dx;

	if (is_target() && vm && budget())
		pr_info("tlbtrace %6lld [%s/%d] compute_tlb vm=%px type=%u tlb_seq=%lld kfd_last=%lld %s\n",
			(long long)atomic64_inc_return(&seqno),
			current->comm, current->pid, vm, ftype,
			avm_tlb_seq(vm), avm_kfd_last_flushed_seq(vm),
			(avm_tlb_seq(vm) == avm_kfd_last_flushed_seq(vm)) ?
				"(seq==last -> WILL SKIP HW flush)" :
				"(seq!=last -> will flush)");
	return 0;
}

/* ---- KFD map/unmap ioctl path (adev, mem, drm_priv) */
static int pre_kfd_map(struct kprobe *p, struct pt_regs *regs)
{
	void *vm = (void *)regs->dx; /* drm_priv ~= vm root; logged as token */
	EV("KFD map_memory_to_gpu drm_priv=%px", vm);
	return 0;
}
static int pre_kfd_unmap(struct kprobe *p, struct pt_regs *regs)
{
	void *vm = (void *)regs->dx;
	EV("KFD unmap_memory_from_gpu drm_priv=%px", vm);
	return 0;
}

/* ---- amdgpu_gem_va_ioctl(dev, data, filp) : graphics render-node VA op */
static int pre_gem_va(struct kprobe *p, struct pt_regs *regs)
{
	EV("GEM gem_va_ioctl (graphics render-node VA map/unmap)");
	return 0;
}

struct probe_desc {
	const char *symbol;
	int (*pre)(struct kprobe *, struct pt_regs *);
	struct kprobe kp;
	bool armed;
};

static struct probe_desc probes[] = {
	{ .symbol = "amdgpu_vm_update_range",                   .pre = pre_update_range },
	{ .symbol = "amdgpu_vm_tlb_fence_create",               .pre = pre_tlb_fence_create },
	{ .symbol = "amdgpu_gmc_flush_gpu_tlb_pasid",           .pre = pre_flush_pasid },
	{ .symbol = "amdgpu_vm_flush_compute_tlb",              .pre = pre_compute_tlb },
	{ .symbol = "amdgpu_amdkfd_gpuvm_map_memory_to_gpu",    .pre = pre_kfd_map },
	{ .symbol = "amdgpu_amdkfd_gpuvm_unmap_memory_from_gpu",.pre = pre_kfd_unmap },
	{ .symbol = "amdgpu_gem_va_ioctl",                      .pre = pre_gem_va },
};

/* Best-effort: read /sys/module/amdgpu/srcversion and compare. */
static int check_srcversion(void)
{
	struct file *f;
	char buf[64] = {0};
	loff_t pos = 0;
	ssize_t n;

	f = filp_open("/sys/module/amdgpu/srcversion", O_RDONLY, 0);
	if (IS_ERR(f)) {
		pr_warn("tlbtrace: cannot open amdgpu srcversion (%ld); skipping guard\n",
			PTR_ERR(f));
		return 0;
	}
	n = kernel_read(f, buf, sizeof(buf) - 1, &pos);
	filp_close(f, NULL);
	if (n <= 0)
		return 0;
	buf[n] = '\0';
	strim(buf);
	if (strcmp(buf, AMDGPU_SRCVERSION_EXPECT) != 0) {
		pr_err("tlbtrace: amdgpu srcversion '%s' != expected '%s' -- struct offsets may be WRONG\n",
		       buf, AMDGPU_SRCVERSION_EXPECT);
		return strict_srcversion ? -EINVAL : 0;
	}
	pr_info("tlbtrace: amdgpu srcversion matches (%s)\n", buf);
	return 0;
}

static int __init tlbtrace_init(void)
{
	int i, armed = 0, rc;

	rc = check_srcversion();
	if (rc)
		return rc;

	for (i = 0; i < ARRAY_SIZE(probes); i++) {
		probes[i].kp.symbol_name = probes[i].symbol;
		probes[i].kp.pre_handler = probes[i].pre;
		rc = register_kprobe(&probes[i].kp);
		if (rc) {
			pr_warn("tlbtrace: register_kprobe(%s) failed: %d (skipping)\n",
				probes[i].symbol, rc);
			probes[i].armed = false;
		} else {
			probes[i].armed = true;
			armed++;
		}
	}
	pr_info("tlbtrace: loaded, %d/%d kprobes armed, targetcomm='%s' maxlog=%d\n",
		armed, (int)ARRAY_SIZE(probes), targetcomm, maxlog);
	if (!armed)
		return -ENODEV;
	return 0;
}

static void __exit tlbtrace_exit(void)
{
	int i;

	for (i = 0; i < ARRAY_SIZE(probes); i++)
		if (probes[i].armed)
			unregister_kprobe(&probes[i].kp);
	pr_info("tlbtrace: unloaded (%lld events emitted)\n",
		(long long)atomic64_read(&seqno));
}

module_init(tlbtrace_init);
module_exit(tlbtrace_exit);
