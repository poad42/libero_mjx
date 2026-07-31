# Out-of-tree build for the gfx1201 mempool TLB-race kernel modules.
#
#   mempool_tlbtrace.ko  - PHASE 1, trace-only kprobe instrumentation
#   mempool_tlbfix.ko    - PHASE 2, behaviour-changing livepatch (forces the
#                          fenced/synchronous GPU TLB flush on the pool
#                          trim -> re-map path so trimming is safe)
#
# Build against the running kernel's build tree (matches vermagic):
#   make
# Load/unload (needs module-load privilege; see README):
#   sudo insmod mempool_tlbtrace.ko
#   sudo rmmod  mempool_tlbtrace
#
# KDIR can be overridden for cross/other-kernel builds.
KDIR ?= /lib/modules/$(shell uname -r)/build
PWD  := $(shell pwd)

obj-m += mempool_tlbtrace.o
obj-m += mempool_tlbfix.o

all:
	$(MAKE) -C $(KDIR) M=$(PWD) modules

clean:
	$(MAKE) -C $(KDIR) M=$(PWD) clean

.PHONY: all clean
