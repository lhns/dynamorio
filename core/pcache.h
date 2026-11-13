/* Lab-only prototype of a translation cache across runs (lhns/v3compat docs/translation-cache.md,
 * Phase 1): finished, mangled basic blocks of never-written image code, replayed through the
 * ordinary emit path. Fixed addresses (ASLR off) apart from DR's vmcode region; not for merge.
 */
#ifndef _PCACHE_H_
#define _PCACHE_H_ 1

/* Opaque: a writer's snapshot of a block, a reader's hit. */
typedef struct _pcache_ref_t pcache_ref_t;
typedef struct _pcache_hit_t pcache_hit_t;

bool
pcache_reading(void);
bool
pcache_writing(void);
bool
pcache_verifying(void);

/* Reader: on a hit fills *ilist, *flags, *vmlist (the vmlist DR's own build would make) and
 * returns the hit, else NULL with nothing allocated.
 */
const pcache_hit_t *
pcache_lookup(dcontext_t *dcontext, app_pc tag, uint initial_flags, bool avx512_before,
              instrlist_t **ilist, uint *flags, void **vmlist);

/* Verify mode: compares fresh, a fresh translation of f's block, with f as installed from the
 * cache (f's ilist, still alive, is installed). Reports and, at -pcache_verify 1, exits 86.
 */
void
pcache_verify(dcontext_t *dcontext, const pcache_hit_t *hit, fragment_t *f, instrlist_t *installed,
              instrlist_t *fresh, uint fresh_flags);

/* Writer: before emit, returns a snapshot of a cacheable block, else NULL. */
pcache_ref_t *
pcache_snapshot(dcontext_t *dcontext, app_pc tag, app_pc end_pc, uint initial_flags,
                bool avx512_before, uint flags, instrlist_t *ilist);
/* Writer: after emit, records the snapshot's block from f's code. Frees the snapshot. */
void
pcache_record(dcontext_t *dcontext, pcache_ref_t *snap, instrlist_t *ilist, fragment_t *f);

/* Process exit: the writer writes its file; both print -pcache_stats. */
void
pcache_process_exit(void);

#endif /* _PCACHE_H_ */
