// SPDX-License-Identifier: GPL-2.0
/*
 * mempool_tlbfix.c - PHASE 2, behaviour-changing live "livepatch" (kprobe +
 * ftrace IPMODIFY) for the gfx1201 hipMallocAsync stream-ordered-pool TLB-flush
 * race. No amdgpu reload; live insmod/rmmod.
 *
 * ============================ Root cause (TRACED) ============================
 * Confirmed live with mempool_tlbtrace.ko on kernel 6.19.0-amd-staging-p2p,
 * amdgpu GC_HWIP 12.0.1 (gfx1201), running rocm_mempool_bug:
 *
 *   - The pool drives the KFD compute-VM path
 *     (amdgpu_amdkfd_gpuvm_map/unmap_memory_to_gpu); those VMs already have
 *     is_compute_context=1, need_tlb_fence=1.
 *   - On the cold re-map, amdgpu_vm_update_range() runs with flush_tlb=0 and the
 *     following amdgpu_vm_flush_compute_tlb() observes tlb_seq==kfd_last_flushed
 *     and SKIPS the HW TLB invalidate. The unmap path sets flush_tlb=1 but the
 *     tlb_seq bump is deferred (async dma_fence callback), so the next map's
 *     check races and skips. The fresh VA is consumed with a stale VA->PA
 *     translation -> stale/zero reads (MEMPOOL_REPORT.md / ASM_BYPASS_REPORT.md).
 *   - The KFD map ioctl calls unreserve_bo_and_vms(&ctx, wait=false): it
 *     attaches the PT-update fence to mem->sync but never waits for it, and the
 *     consuming compute queue does not honour that fence -> nothing serialises
 *     the re-map's TLB flush before the stream's fill/kernel.
 *
 * ============================ Fix (mode=syncmap) =============================
 * The pool's trim/re-map is driven OVERWHELMINGLY by the GEM VA ioctl path
 * (amdgpu_gem_va_ioctl -> amdgpu_gem_va_update_vm), NOT the KFD ioctls. A live
 * ftrace stacktrace of amdgpu_vm_update_range during the reproducer shows
 * ~1600 GEM-VA hits vs ~517 KFD-ioctl hits per burst. amdgpu_gem_va_ioctl
 * returns to userspace WITHOUT waiting on the fence it gets back, so the
 * stream's consuming kernel can run before the re-map's TLB flush completes.
 *
 * Three coordinated live hooks:
 *
 *   (A) kprobe @ amdgpu_vm_update_range entry: for the pool's compute VMs, force
 *       need_tlb_fence=true (so amdgpu_vm_tlb_flush takes the
 *       amdgpu_vm_tlb_fence_create() path) AND force flush_tlb=1 (5th SysV arg,
 *       %r8). This makes amdgpu run its correctly-ordered fenced flush:
 *       amdgpu_vm_tlb_fence_create -> amdgpu_tlb_fence_work, which
 *       dma_fence_wait()s for the PT-update to LAND, then issues a heavyweight
 *       all-hub amdgpu_gmc_flush_gpu_tlb_pasid(adev,pasid,2,true,0). The
 *       resulting HW-flush-COMPLETION fence becomes *fence -> bo_va->last_pt_update
 *       and vm->last_update.
 *
 *   (B) ftrace IPMODIFY wrapper @ amdgpu_gem_va_update_vm (the dominant path):
 *       after the original runs, dma_fence_wait_timeout() on the fence it
 *       RETURNS -- the merge of vm->last_update and bo_va->last_pt_update, i.e.
 *       the HW-flush-completion fence from (A). This blocks the GEM VA ioctl
 *       (and thus the stream's consuming kernel) until the new translation is
 *       coherent on the GPU. NOTE: vm->last_tlb_flush is the WRONG fence to wait
 *       on -- amdgpu_vm_tlb_flush assigns it the PT-write fence BEFORE
 *       tlb_fence_create swaps in the flush-completion fence.
 *
 *   (C) ftrace IPMODIFY wrappers @ amdgpu_amdkfd_gpuvm_{,un}map_memory_to/from_gpu:
 *       after the original returns, amdgpu_amdkfd_gpuvm_sync_memory(adev, mem,
 *       false) waits on mem->sync (= bo_va->last_pt_update = the flush-completion
 *       fence). Covers the minority KFD path symmetrically.
 *
 * Why the cheaper attempts failed (measured, for the record):
 *   mode=remapflush (A only)            : kills the catastrophic mode but leaves
 *                                         a ~100-200/proc residual floor (flush
 *                                         created, never waited on).
 *   syncmap waiting on the KFD ioctls   : still ~130/proc -- those ioctls are
 *                                         only ~13% of remaps.
 *   syncmap waiting on vm->last_tlb_flush: still ~130/proc -- wrong fence (PT
 *                                         write, not flush completion).
 *   syncmap waiting on the GEM-VA returned fence (B): 0 corruptions.
 *
 * VALIDATED (trimming ACTIVE, release threshold = DEFAULT, gfx1201/GPU3):
 *   baseline (no module) : 0/20 cold procs clean, 53202 corrupt iters total.
 *   patched (syncmap)    : 15/15 cold procs clean, 0 corrupt iters. dmesg clean.
 *
 * Adding TLB flushes / waiting is correctness-safe: it costs a little
 * performance, never causes corruption. Set onlycomm=<substr> to confine the
 * behaviour change to matching processes (used to isolate co-tenant GPUs).
 * x86-64 only.
 */
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/kprobes.h>
#include <linux/ftrace.h>
#include <linux/sched.h>
#include <linux/string.h>
#include <linux/atomic.h>
#include <linux/fs.h>
#include <linux/rcupdate.h>
#include <linux/kallsyms.h>
#include <linux/jiffies.h>
#include <asm/ptrace.h>

#include "amdgpu_vm_offsets.h"

/*
 * Hooks (A) read/write the 5th SysV arg in %r8 and args in %rsi/%rdx/%rcx.
 * That register mapping is x86-64-specific; building elsewhere would silently
 * poke the wrong registers (potential corruption), so refuse at compile time.
 */
#ifndef __x86_64__
#error "mempool_tlbfix relies on the x86-64 SysV register layout; x86-64 only"
#endif

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Agent-Kmod");
MODULE_DESCRIPTION("Live kprobe+ftrace fix for the gfx1201 mempool TLB-flush race");
MODULE_VERSION("3.0");

static char *mode = "syncmap";
module_param(mode, charp, 0644);
MODULE_PARM_DESC(mode, "syncmap (default) | remapflush | syncflush | both | fence");

static char *onlycomm = "";
module_param(onlycomm, charp, 0644);
MODULE_PARM_DESC(onlycomm, "if set, only patch VMs/flushes driven by processes whose comm contains this substring");

static int force_all_hub = 1;
module_param(force_all_hub, int, 0644);
MODULE_PARM_DESC(force_all_hub, "syncflush mode: force gmc_flush_gpu_tlb_pasid all_hub=1 (default 1)");

static int strict_srcversion = 1;
module_param(strict_srcversion, int, 0644);
MODULE_PARM_DESC(strict_srcversion, "refuse to load if amdgpu srcversion mismatches (default 1)");

/*
 * Upper bound (ms) for each re-map TLB-flush-completion fence wait. 0 keeps the
 * validated behaviour: an indefinite, uninterruptible wait that mirrors the
 * driver's own dma_fence_wait() OOM-fallback under the same locks. A finite
 * value lets an operator avoid a wedged D-state task if the GPU hangs (the
 * flush normally lands in microseconds); on timeout we warn but proceed, since
 * a multi-second stall here means the GPU is already in TDR/reset territory.
 */
static int wait_ms;
module_param(wait_ms, int, 0644);
MODULE_PARM_DESC(wait_ms, "max ms to wait on each re-map TLB-flush fence; 0 = indefinite (default, validated)");

static atomic64_t n_force_flush = ATOMIC64_INIT(0);
static atomic64_t n_all_hub = ATOMIC64_INIT(0);
static atomic64_t n_fence_forced = ATOMIC64_INIT(0);
static atomic64_t n_remap_forced = ATOMIC64_INIT(0);
static atomic64_t n_map_waited = ATOMIC64_INIT(0);
static atomic64_t n_unmap_waited = ATOMIC64_INIT(0);
static atomic64_t n_gem_waited = ATOMIC64_INIT(0);

static int do_remap, do_sync, do_fence, do_syncmap;

static bool comm_ok(void)
{
	if (!onlycomm || !onlycomm[0])
		return true;
	return strstr(current->comm, onlycomm) != NULL;
}

/* ---- resolve any kallsyms symbol address via a throwaway kprobe ---- */
static unsigned long resolve_sym(const char *name)
{
	struct kprobe kp = { .symbol_name = name };
	unsigned long addr = 0;

	if (register_kprobe(&kp) == 0) {
		addr = (unsigned long)kp.addr;
		unregister_kprobe(&kp);
	}
	return addr;
}

/* ====================== (A) kprobe handlers ======================= */

/* amdgpu_vm_flush_compute_tlb(adev, vm, flush_type, xcc_mask) - syncflush */
static int pre_compute_tlb(struct kprobe *p, struct pt_regs *regs)
{
	void *vm = (void *)regs->si;

	if (!vm || !comm_ok())
		return 0;
	atomic64_set((atomic64_t *)((char *)vm + AVM_OFF_KFD_LAST_FLUSHED_SEQ),
		     avm_tlb_seq(vm) - 1);
	atomic64_inc(&n_force_flush);
	return 0;
}

/* amdgpu_gmc_flush_gpu_tlb_pasid(adev, pasid, flush_type, all_hub, inst) */
static int pre_flush_pasid(struct kprobe *p, struct pt_regs *regs)
{
	if (!comm_ok())
		return 0;
	regs->dx = 2;	/* heavyweight */
	regs->cx = 1;	/* all_hub */
	atomic64_inc(&n_all_hub);
	return 0;
}

/* amdgpu_vm_update_range(adev, vm, immediate, unlocked, flush_tlb, ...) */
static int pre_update_range(struct kprobe *p, struct pt_regs *regs)
{
	void *vm = (void *)regs->si;

	if (!vm || !comm_ok())
		return 0;
	/*
	 * syncmap needs BOTH: (1) need_tlb_fence=true so amdgpu_vm_tlb_flush()
	 * takes the amdgpu_vm_tlb_fence_create() path that REPLACES *fence (->
	 * bo_va->last_pt_update -> mem->sync) with the real HW-flush-completion
	 * fence; and (2) flush_tlb=1 so that fenced flush is actually engaged on
	 * this re-map. Without (1), forcing (2) only bumps tlb_seq (deferred
	 * flush) and last_pt_update stays the PT-write fence -> map-wait blocks
	 * on the wrong fence and the HW flush leaks into the racy window.
	 */
	if ((do_fence || do_syncmap) && avm_is_compute(vm) &&
	    !avm_need_tlb_fence(vm)) {
		avm_set_need_tlb_fence(vm, true);
		atomic64_inc(&n_fence_forced);
	}
	if ((do_remap || do_syncmap) && avm_is_compute(vm) &&
	    ((regs->r8 & 0xff) == 0)) {
		regs->r8 = 1;	/* flush_tlb = true -> fenced flush */
		atomic64_inc(&n_remap_forced);
	}
	return 0;
}

static struct kprobe kp_compute = {
	.symbol_name = "amdgpu_vm_flush_compute_tlb",
	.pre_handler = pre_compute_tlb,
};
static struct kprobe kp_gmc = {
	.symbol_name = "amdgpu_gmc_flush_gpu_tlb_pasid",
	.pre_handler = pre_flush_pasid,
};
static struct kprobe kp_update = {
	.symbol_name = "amdgpu_vm_update_range",
	.pre_handler = pre_update_range,
};
static bool armed_compute, armed_gmc, armed_update;

/* ====================== (B) ftrace IPMODIFY wrapper ======================= */

typedef int (*map_fn_t)(void *adev, void *mem, void *drm_priv);
typedef int (*syncmem_fn_t)(void *adev, void *mem, bool intr);
typedef void *(*gemupd_fn_t)(void *adev, void *vm, void *bo_va, u32 operation);
typedef long (*fencewait_fn_t)(void *fence, bool intr, long timeout);

static unsigned long addr_map;            /* amdgpu_amdkfd_gpuvm_map_memory_to_gpu */
static unsigned long addr_unmap;          /* amdgpu_amdkfd_gpuvm_unmap_memory_from_gpu */
static unsigned long addr_gemupd;         /* amdgpu_gem_va_update_vm */
static syncmem_fn_t fn_sync_memory;       /* amdgpu_amdkfd_gpuvm_sync_memory */
static fencewait_fn_t fn_fence_wait;      /* dma_fence_wait_timeout */

/* Wait on a re-map's flush-completion fence; bound by wait_ms (0 = forever). */
static long notrace tlbfix_fence_wait(void *fence)
{
	long timeout = (wait_ms > 0) ? msecs_to_jiffies(wait_ms) : MAX_SCHEDULE_TIMEOUT;
	long r = fn_fence_wait(fence, false, timeout);

	if (r == 0)
		pr_warn_ratelimited("tlbfix: re-map TLB-flush fence wait timed out after %d ms; GPU may be wedged\n",
				    wait_ms);
	return r;
}

/*
 * Replacement for map: run the original, then wait for the (forced) fenced TLB
 * flush to complete via mem->sync, so the map ioctl serialises against it.
 */
static notrace int hooked_map(void *adev, void *mem, void *drm_priv)
{
	map_fn_t orig = (map_fn_t)addr_map;
	int ret = orig(adev, mem, drm_priv);

	if (ret == 0 && comm_ok() && fn_sync_memory) {
		fn_sync_memory(adev, mem, false);
		atomic64_inc(&n_map_waited);
	}
	return ret;
}

/*
 * Replacement for unmap (the pool's trim path): run the original, then wait on
 * mem->sync. unmap_memory_from_gpu -> unreserve_bo_and_vms() runs
 * amdgpu_vm_clear_freed() (flush_tlb=true) and stashes the resulting fence in
 * mem->sync but NEVER waits. Without this, the trim's TLB flush is deferred and
 * the physical page can be recycled into a new VA before the old translation is
 * invalidated -> stale read. Waiting here closes the page-reuse window.
 */
static notrace int hooked_unmap(void *adev, void *mem, void *drm_priv)
{
	map_fn_t orig = (map_fn_t)addr_unmap;
	int ret = orig(adev, mem, drm_priv);

	if (ret == 0 && comm_ok() && fn_sync_memory) {
		fn_sync_memory(adev, mem, false);
		atomic64_inc(&n_unmap_waited);
	}
	return ret;
}

/*
 * Replacement for amdgpu_gem_va_update_vm -- THE dominant pool remap path.
 * Tracing showed the hipMallocAsync pool's trim/re-map is driven by GEM VA
 * ioctls (amdgpu_gem_va_ioctl -> amdgpu_gem_va_update_vm), not the KFD ioctls,
 * and amdgpu_gem_va_ioctl returns to userspace WITHOUT waiting on the fence it
 * gets back.
 *
 * The fence to wait on is the one this function RETURNS: a merge of
 * vm->last_update and bo_va->last_pt_update. With (A) forcing the fenced flush
 * (need_tlb_fence + flush_tlb=1), amdgpu_vm_tlb_flush() ->
 * amdgpu_vm_tlb_fence_create() replaces *fence (which becomes both last_update
 * and last_pt_update) with the HW-flush-COMPLETION fence. NOTE: vm->last_tlb_flush
 * is NOT that fence -- it is assigned the PT-write fence *before* tlb_fence_create
 * runs, so it only signals that the page-table write landed, not that the GPU TLB
 * flush completed. We must wait on the returned merged fence. Waiting here blocks
 * the VA ioctl -- and thus the stream's consuming kernel -- until the new
 * translation is coherent on the GPU. The driver itself dma_fence_wait()s inside
 * this function (OOM fallback) under the same locks, so this is deadlock-safe.
 */
static notrace void *hooked_gemupd(void *adev, void *vm, void *bo_va, u32 op)
{
	gemupd_fn_t orig = (gemupd_fn_t)addr_gemupd;
	void *fence = orig(adev, vm, bo_va, op);

	if (fence && comm_ok() && vm && avm_is_compute(vm) && fn_fence_wait) {
		tlbfix_fence_wait(fence);
		atomic64_inc(&n_gem_waited);
	}
	return fence;
}

/*
 * Self-contained recursion guard. When the thunk redirects into hooked_*,
 * hooked_* calls the original entry (addr_*) which is still ftrace-hooked,
 * re-entering the thunk with parent_ip pointing back inside hooked_*.
 * within_module() is inlined on this kernel (absent from kallsyms), so instead
 * we detect re-entry by checking parent_ip against the replacement's own code
 * range. Each hooked_* is the only caller of the hooked function inside this
 * module's text, so a parent_ip inside hooked_X means "re-entry, pass through".
 *
 * The exact byte size of each hooked_X is resolved at load via
 * kallsyms_lookup_size_offset() (these static fns are in the module's kallsyms),
 * which is robust to compiler layout changes. HOOKED_SPAN is the fallback if the
 * lookup ever fails; measured sizes are ~150-160 bytes, far under it.
 */
#define HOOKED_SPAN 0x400UL

static unsigned long span_map = HOOKED_SPAN;
static unsigned long span_unmap = HOOKED_SPAN;
static unsigned long span_gemupd = HOOKED_SPAN;

/*
 * kallsyms_lookup_size_offset() is exported to the kernel but NOT to modules
 * (absent from Module.symvers), so we resolve it the same way as the other
 * non-exported symbols this module uses: a throwaway kprobe in resolve_sym().
 */
typedef int (*ksymsize_fn_t)(unsigned long addr, unsigned long *symsize,
			     unsigned long *offset);
static ksymsize_fn_t fn_ksym_size;

static unsigned long resolve_fn_span(void *fn)
{
	unsigned long size = 0, off = 0;

	if (fn_ksym_size &&
	    fn_ksym_size((unsigned long)fn, &size, &off) && size)
		return size;
	return HOOKED_SPAN;
}

static bool notrace in_replacement(unsigned long ip, void *fn, unsigned long span)
{
	unsigned long h = (unsigned long)fn;

	return ip >= h && ip < h + span;
}

static void notrace thunk_map(unsigned long ip, unsigned long parent_ip,
			      struct ftrace_ops *ops, struct ftrace_regs *fregs)
{
	struct pt_regs *regs;

	if (in_replacement(parent_ip, hooked_map, span_map))
		return;
	regs = ftrace_get_regs(fregs);
	if (regs)
		regs->ip = (unsigned long)hooked_map;
}

static void notrace thunk_unmap(unsigned long ip, unsigned long parent_ip,
				struct ftrace_ops *ops, struct ftrace_regs *fregs)
{
	struct pt_regs *regs;

	if (in_replacement(parent_ip, hooked_unmap, span_unmap))
		return;
	regs = ftrace_get_regs(fregs);
	if (regs)
		regs->ip = (unsigned long)hooked_unmap;
}

static void notrace thunk_gemupd(unsigned long ip, unsigned long parent_ip,
				 struct ftrace_ops *ops, struct ftrace_regs *fregs)
{
	struct pt_regs *regs;

	if (in_replacement(parent_ip, hooked_gemupd, span_gemupd))
		return;
	regs = ftrace_get_regs(fregs);
	if (regs)
		regs->ip = (unsigned long)hooked_gemupd;
}

static struct ftrace_ops map_ops = {
	.func = thunk_map,
	.flags = FTRACE_OPS_FL_SAVE_REGS | FTRACE_OPS_FL_IPMODIFY |
		 FTRACE_OPS_FL_RECURSION,
};
static struct ftrace_ops unmap_ops = {
	.func = thunk_unmap,
	.flags = FTRACE_OPS_FL_SAVE_REGS | FTRACE_OPS_FL_IPMODIFY |
		 FTRACE_OPS_FL_RECURSION,
};
static struct ftrace_ops gemupd_ops = {
	.func = thunk_gemupd,
	.flags = FTRACE_OPS_FL_SAVE_REGS | FTRACE_OPS_FL_IPMODIFY |
		 FTRACE_OPS_FL_RECURSION,
};
static bool armed_map_ftrace, armed_unmap_ftrace, armed_gemupd_ftrace;

static int arm_one(struct ftrace_ops *ops, unsigned long addr, const char *what)
{
	int rc = ftrace_set_filter_ip(ops, addr, 0, 0);

	if (rc) {
		pr_err("tlbfix: ftrace_set_filter_ip(%s) failed: %d\n", what, rc);
		return rc;
	}
	rc = register_ftrace_function(ops);
	if (rc) {
		pr_err("tlbfix: register_ftrace_function(%s) failed: %d\n", what, rc);
		ftrace_set_filter_ip(ops, addr, 1, 0);
		return rc;
	}
	return 0;
}

static int install_ftrace_wrapper(void)
{
	int rc;

	addr_map = resolve_sym("amdgpu_amdkfd_gpuvm_map_memory_to_gpu");
	addr_unmap = resolve_sym("amdgpu_amdkfd_gpuvm_unmap_memory_from_gpu");
	addr_gemupd = resolve_sym("amdgpu_gem_va_update_vm");
	fn_sync_memory = (syncmem_fn_t)resolve_sym("amdgpu_amdkfd_gpuvm_sync_memory");
	fn_fence_wait = (fencewait_fn_t)resolve_sym("dma_fence_wait_timeout");
	if (!addr_map || !addr_unmap || !addr_gemupd || !fn_sync_memory ||
	    !fn_fence_wait) {
		pr_err("tlbfix: symbol resolve failed (map=%lx unmap=%lx gemupd=%lx sync=%px wait=%px)\n",
		       addr_map, addr_unmap, addr_gemupd, fn_sync_memory,
		       fn_fence_wait);
		return -ENOENT;
	}

	/* Exact re-entry-guard spans (fallback HOOKED_SPAN if lookup fails). */
	fn_ksym_size = (ksymsize_fn_t)resolve_sym("kallsyms_lookup_size_offset");
	span_map = resolve_fn_span(hooked_map);
	span_unmap = resolve_fn_span(hooked_unmap);
	span_gemupd = resolve_fn_span(hooked_gemupd);
	pr_info("tlbfix: reentry-guard spans map=%lu unmap=%lu gemupd=%lu (fallback %lu)\n",
		span_map, span_unmap, span_gemupd, HOOKED_SPAN);

	rc = arm_one(&map_ops, addr_map, "map");
	if (rc)
		return rc;
	armed_map_ftrace = true;

	rc = arm_one(&unmap_ops, addr_unmap, "unmap");
	if (rc)
		return rc;
	armed_unmap_ftrace = true;

	rc = arm_one(&gemupd_ops, addr_gemupd, "gemupd");
	if (rc)
		return rc;
	armed_gemupd_ftrace = true;
	return 0;
}

static void remove_ftrace_wrapper(void)
{
	bool any = armed_map_ftrace || armed_unmap_ftrace || armed_gemupd_ftrace;

	if (armed_map_ftrace) {
		unregister_ftrace_function(&map_ops);
		ftrace_set_filter_ip(&map_ops, addr_map, 1, 0);
		armed_map_ftrace = false;
	}
	if (armed_unmap_ftrace) {
		unregister_ftrace_function(&unmap_ops);
		ftrace_set_filter_ip(&unmap_ops, addr_unmap, 1, 0);
		armed_unmap_ftrace = false;
	}
	if (armed_gemupd_ftrace) {
		unregister_ftrace_function(&gemupd_ops);
		ftrace_set_filter_ip(&gemupd_ops, addr_gemupd, 1, 0);
		armed_gemupd_ftrace = false;
	}
	/* Make sure no task is still executing inside hooked_* before unload. */
	if (any)
		synchronize_rcu_tasks();
}

/* ============================== srcversion guard ========================== */
static int check_srcversion(void)
{
	struct file *f;
	char buf[64] = {0};
	loff_t pos = 0;
	ssize_t n;

	f = filp_open("/sys/module/amdgpu/srcversion", O_RDONLY, 0);
	if (IS_ERR(f)) {
		pr_warn("tlbfix: cannot open amdgpu srcversion (%ld); skipping guard\n",
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
		pr_err("tlbfix: amdgpu srcversion '%s' != expected '%s' -- refusing\n",
		       buf, AMDGPU_SRCVERSION_EXPECT);
		return strict_srcversion ? -EINVAL : 0;
	}
	pr_info("tlbfix: amdgpu srcversion matches (%s)\n", buf);
	return 0;
}

static int __init tlbfix_init(void)
{
	int rc;

	do_syncmap = (strcmp(mode, "syncmap") == 0);
	do_remap   = (strcmp(mode, "remapflush") == 0) || (strcmp(mode, "both") == 0);
	do_sync    = (strcmp(mode, "syncflush") == 0)  || (strcmp(mode, "both") == 0);
	do_fence   = (strcmp(mode, "fence") == 0);

	if (!do_syncmap && !do_remap && !do_sync && !do_fence) {
		pr_err("tlbfix: unknown mode '%s'\n", mode);
		return -EINVAL;
	}

	rc = check_srcversion();
	if (rc)
		return rc;

	/* (A) update_range kprobe: needed for remap/syncmap/fence. */
	if (do_remap || do_syncmap || do_fence) {
		rc = register_kprobe(&kp_update);
		if (rc) {
			pr_err("tlbfix: register_kprobe(amdgpu_vm_update_range) failed: %d\n", rc);
			return rc;
		}
		armed_update = true;
	}

	/* (B) ftrace wait-wrapper for syncmap. */
	if (do_syncmap) {
		rc = install_ftrace_wrapper();
		if (rc)
			goto err;
	}

	/* syncflush extras. */
	if (do_sync) {
		rc = register_kprobe(&kp_compute);
		if (rc) {
			pr_err("tlbfix: register_kprobe(amdgpu_vm_flush_compute_tlb) failed: %d\n", rc);
			goto err;
		}
		armed_compute = true;
		if (force_all_hub) {
			if (register_kprobe(&kp_gmc) == 0)
				armed_gmc = true;
			else
				pr_warn("tlbfix: gmc kprobe failed; no all_hub forcing\n");
		}
	}

	pr_info("tlbfix: loaded mode=%s onlycomm='%s' wait_ms=%d [remap_flush=%s gem_wait=%s map_wait=%s unmap_wait=%s compute_flush=%s all_hub=%s need_fence=%s]\n",
		mode, onlycomm, wait_ms,
		((do_remap || do_syncmap) && armed_update) ? "ON" : "off",
		armed_gemupd_ftrace ? "ON" : "off",
		armed_map_ftrace ? "ON" : "off",
		armed_unmap_ftrace ? "ON" : "off",
		armed_compute ? "ON" : "off",
		armed_gmc ? "ON" : "off",
		((do_fence || do_syncmap) && armed_update) ? "ON" : "off");
	return 0;

err:
	remove_ftrace_wrapper();
	if (armed_compute)
		unregister_kprobe(&kp_compute);
	if (armed_gmc)
		unregister_kprobe(&kp_gmc);
	if (armed_update)
		unregister_kprobe(&kp_update);
	return rc;
}

static void __exit tlbfix_exit(void)
{
	remove_ftrace_wrapper();
	if (armed_compute)
		unregister_kprobe(&kp_compute);
	if (armed_gmc)
		unregister_kprobe(&kp_gmc);
	if (armed_update)
		unregister_kprobe(&kp_update);
	pr_info("tlbfix: unloaded (remap_flush forced %lld, gem waits %lld, map waits %lld, unmap waits %lld, compute flushes %lld, all-hub %lld, need_fence %lld)\n",
		(long long)atomic64_read(&n_remap_forced),
		(long long)atomic64_read(&n_gem_waited),
		(long long)atomic64_read(&n_map_waited),
		(long long)atomic64_read(&n_unmap_waited),
		(long long)atomic64_read(&n_force_flush),
		(long long)atomic64_read(&n_all_hub),
		(long long)atomic64_read(&n_fence_forced));
}

module_init(tlbfix_init);
module_exit(tlbfix_exit);
