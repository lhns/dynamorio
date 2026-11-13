/* LAB ONLY: per-thread event history and a ledger of freed shared fragments, printed when
 * DynamoRIO aborts on a fault, to tell whether the faulting pc is a freed fragment or stub.
 */
#ifndef V3DIAG_H
#define V3DIAG_H

enum {
    V3D_ENTER = 1, /* a=tag b=start_pc c=flags u=flushtime_last_update */
    V3D_DISPATCH,  /* a=last_exit b=next_tag c=last_fragment u=whereami */
    V3D_FLUSHTIME, /* a=old b=new */
    V3D_SIG,       /* a=sig b=pc c=whereami */
    V3D_SIGRET,    /* a=next_tag */
    V3D_SIGPEND,   /* a=sig b=pc c=fragment start (the one unlinked for it) */
};

typedef struct {
    uint64 seq;
    uint kind, u;
    ptr_uint_t a, b, c;
} v3d_ev_t;

#define V3D_THREADS 1024
#define V3D_EVENTS 48
typedef struct {
    thread_id_t tid;
    uint pos;
    v3d_ev_t ev[V3D_EVENTS];
} v3d_thread_t;

#define V3D_FREES (1 << 17)
#define V3D_STUBS 6
typedef struct {
    uint64 seq;
    app_pc tag;
    byte *start;
    uint size, flags, ft_global, ft_self, nstubs;
    thread_id_t tid;
    byte *stub[V3D_STUBS];
} v3d_free_t;

extern v3d_thread_t v3d_threads[V3D_THREADS];
extern v3d_free_t v3d_frees[V3D_FREES];
extern volatile uint64 v3d_seq, v3d_nfree;

static inline void
v3d_log(dcontext_t *dc, uint kind, uint u, ptr_uint_t a, ptr_uint_t b, ptr_uint_t c)
{
    if (dc == NULL || dc == GLOBAL_DCONTEXT)
        return;
    v3d_thread_t *t = &v3d_threads[(uint)dc->owning_thread % V3D_THREADS];
    if (t->tid != dc->owning_thread) {
        t->tid = dc->owning_thread;
        t->pos = 0;
    }
    v3d_ev_t *e = &t->ev[t->pos++ % V3D_EVENTS];
    e->seq = __atomic_fetch_add(&v3d_seq, 1, __ATOMIC_RELAXED);
    e->kind = kind;
    e->u = u;
    e->a = a;
    e->b = b;
    e->c = c;
}

#endif
