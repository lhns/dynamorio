/* Lab-only prototype of a translation cache across runs: see pcache.h.
 *
 * An entry is a finished basic block (after the client and mangling) recorded as a short
 * instruction list: raw byte runs, app instructions with a rip-relative field (replayed from
 * the app bytes, as DR builds them), DR/client instructions with a pc-relative field (target
 * stored), intra-block jumps by index, and symbolic exit ctis. A hit replays that list through
 * emit_fragment_ex(), so padding, stubs, linkstubs, vmarea registration, linking and the
 * fragment table are DR's ordinary code at the real address.
 *
 * A hit needs: the file's key (DR and private module builds and bases, options, CPUID, XCR0);
 * the same initial flags and AVX-512 state; the module's identity; the range in never-written
 * image areas (VM_UNMOD_IMAGE); and the app bytes equal to the recorded ones.
 */
#include "globals.h"
#include "fragment.h"
#include "link.h"
#include "emit.h"
#include "fcache.h"
#include "vmareas.h"
#include "instrlist.h"
#include "module_shared.h"
#include "heap.h"
#include "arch.h"
#include "instr.h"
#include "instr_create_shared.h"
#include "decode.h"
#include "disassemble.h"
#include "pcache.h"
#include <elf.h>

#define PC_MAGIC 0x3165686361637076ULL /* "vpcache1" */
#define PC_VERSION 1
#define MAX_SRC 4096
#define MAX_INSTRS 2048
/* vmareas.c's private VM_WRITABLE and VM_UNMOD_IMAGE */
#define PC_VM_WRITABLE 0x1
#define PC_VM_UNMOD_IMAGE 0x2

enum { R_RAW = 1, R_APPREL, R_REL, R_LABEL, R_ICTI, R_EXIT };

typedef struct {
    uint64 magic;
    uint version;
    uint num;
    uint64 key;
    uint64 vm_lo, vm_hi; /* the writer's vmcode region, also in the key */
    uint64 index_off, data_off, data_len;
} pc_header_t;

typedef struct {
    uint64 tag;
    uint64 off; /* into data */
} pc_index_t;

typedef struct {
    uint64 tag;
    uint64 modid;
    uint64 mod_start;
    uint initial_flags;
    uint flags;
    ushort src_len;
    ushort nrec;
    ushort avx512;
    ushort pad;
    uint rec_len;
    /* then src bytes, then records */
} pc_entry_t;

struct _pcache_ref_t {
    app_pc tag, end;
    uint initial_flags, flags;
    bool avx512;
    uint n;
    struct snap_ins {
        instr_t *in;
        byte kind; /* R_* or 0 for a non-emitted, non-target label */
        bool is_target;
        ushort opcode;
        uint prefixes, xflags;
        app_pc target;
        int tgt;  /* snapshot index of an ICTI's target */
        int pos;  /* pc-relative field of an APPREL/REL */
        uint rec; /* record index */
    } * ins;
};

static int mode = -1; /* 0 off, 1 read, 2 write */
static bool key_done;
static uint64 run_key;
static byte *vm_lo, *vm_hi;

/* reader */
static const pc_header_t *hdr;
static const pc_index_t *index_;
static const byte *data;

/* writer: entries kept in DR heap until exit */
static volatile int wlock;
static byte **wents;
static uint wnum, wcap;
static bool written;

/* stats */
static uint s_lookup, s_hit, s_noentry, s_iflags, s_avx, s_module, s_area, s_bytes, s_build;
static uint s_vok, s_vbad, s_rec, s_skip[8];
static uint64 t_snap, t_rec, t_look, t_write; /* rdtsc cycles */

static inline uint64
tsc(void)
{
    uint lo, hi;
    __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
    return ((uint64)hi << 32) | lo;
}

static uint64
fnv(uint64 h, const void *p, size_t n)
{
    const byte *b = (const byte *)p;
    if (h == 0)
        h = 0xcbf29ce484222325ULL;
    for (size_t i = 0; i < n; i++) {
        h ^= b[i];
        h *= 0x100000001b3ULL;
    }
    return h;
}

static void
spin_lock(void)
{
    while (!atomic_compare_exchange_int(&wlock, 0, 1))
        os_thread_yield();
}

static void
spin_unlock(void)
{
    ATOMIC_4BYTE_WRITE(&wlock, 0, false);
}

/* ELF build-id at a mapped image, else a hash of its ELF and program headers. */
static uint64
elf_identity(app_pc base)
{
    Elf64_Ehdr eh;
    Elf64_Phdr ph[64];
    if (base == NULL || !d_r_safe_read(base, sizeof(eh), &eh) ||
        memcmp(eh.e_ident, ELFMAG, SELFMAG) != 0 || eh.e_phnum > 64 ||
        !d_r_safe_read(base + eh.e_phoff, eh.e_phnum * sizeof(Elf64_Phdr), ph))
        return 0;
    ptr_uint_t min_vaddr = (ptr_uint_t)-1;
    for (int i = 0; i < eh.e_phnum; i++) {
        if (ph[i].p_type == PT_LOAD && ph[i].p_vaddr < min_vaddr)
            min_vaddr = ph[i].p_vaddr;
    }
    app_pc bias = base - ALIGN_BACKWARD(min_vaddr, PAGE_SIZE);
    for (int i = 0; i < eh.e_phnum; i++) {
        byte note[1024];
        if (ph[i].p_type != PT_NOTE || ph[i].p_filesz > sizeof(note) ||
            !d_r_safe_read(bias + ph[i].p_vaddr, ph[i].p_filesz, note))
            continue;
        for (size_t o = 0; o + sizeof(Elf64_Nhdr) <= ph[i].p_filesz;) {
            Elf64_Nhdr *nh = (Elf64_Nhdr *)(note + o);
            size_t name_off = o + sizeof(*nh);
            size_t desc_off = name_off + ALIGN_FORWARD(nh->n_namesz, 4);
            size_t next = desc_off + ALIGN_FORWARD(nh->n_descsz, 4);
            if (next > ph[i].p_filesz)
                break;
            if (nh->n_type == NT_GNU_BUILD_ID && nh->n_namesz == 4 &&
                memcmp(note + name_off, "GNU", 4) == 0)
                return fnv(0, note + desc_off, nh->n_descsz) | 1;
            o = next;
        }
    }
    return fnv(fnv(0, &eh, sizeof(eh)), ph, eh.e_phnum * sizeof(Elf64_Phdr)) & ~1ULL;
}

static void
cpu_key(uint64 *h)
{
    uint r[4];
    static const uint leaves[][2] = { { 1, 0 }, { 7, 0 }, { 7, 1 }, { 0xd, 0 }, { 0x80000001, 0 } };
    for (uint i = 0; i < sizeof(leaves) / sizeof(leaves[0]); i++) {
        __asm__ __volatile__("cpuid"
                             : "=a"(r[0]), "=b"(r[1]), "=c"(r[2]), "=d"(r[3])
                             : "a"(leaves[i][0]), "c"(leaves[i][1]));
        if (leaves[i][0] == 1)
            r[1] = 0; /* APIC id, CLFLUSH size, logical count */
        *h = fnv(*h, r, sizeof(r));
    }
    uint lo, hi;
    __asm__ __volatile__("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
    *h = fnv(*h, &lo, sizeof(lo));
    *h = fnv(*h, &hi, sizeof(hi));
}

static void
compute_key(void)
{
    uint64 h = fnv(0, "v3pcache", 8), v;
    /* DR picks rip-relative or absolute forms, and direct or indirect calls, by reach from the
     * whole vmcode region: every block depends on where it is, and on the random offset DR gives
     * it unless -vm_max_offset is 0.
     */
    vm_lo = vmcode_get_start();
    vm_hi = vmcode_get_end();
    h = fnv(h, &vm_lo, sizeof(vm_lo));
    h = fnv(h, &vm_hi, sizeof(vm_hi));
    v = elf_identity(get_dynamorio_dll_start());
    h = fnv(h, &v, sizeof(v));
    v = (uint64)get_dynamorio_dll_start();
    h = fnv(h, &v, sizeof(v));
    acquire_recursive_lock(&privload_lock);
    for (privmod_t *m = privload_first_module(); m != NULL; m = privload_next_module(m)) {
        if (m->externally_loaded)
            continue;
        h = fnv(h, m->name, strlen(m->name));
        v = elf_identity(m->base);
        h = fnv(h, &v, sizeof(v));
        v = (uint64)m->base;
        h = fnv(h, &v, sizeof(v));
        if (DYNAMO_OPTION(pcache_stats) > 1)
            print_file(STDERR, "pcache: key mod %s " PFX " %llx\n", m->name, m->base, v);
    }
    release_recursive_lock(&privload_lock);
    /* Every option, the client's included, but the cache's own. */
    options_t *o = (options_t *)global_heap_alloc(sizeof(*o) HEAPACCT(ACCT_OTHER));
    char *s = (char *)global_heap_alloc(MAX_OPTIONS_STRING HEAPACCT(ACCT_OTHER));
    *o = dynamo_options;
    o->pcache_mode = 0;
    o->pcache_verify = 0;
    o->pcache_stats = 0;
    o->pcache_dir[0] = '\0';
    get_dynamo_options_string(o, s, MAX_OPTIONS_STRING, true);
    h = fnv(h, s, strlen(s));
    if (DYNAMO_OPTION(pcache_stats) > 1) {
        print_file(STDERR, "pcache: key dr " PFX " vm " PFX "-" PFX " opts %s\n",
                   get_dynamorio_dll_start(), vm_lo, vm_hi, s);
    }
    global_heap_free(s, MAX_OPTIONS_STRING HEAPACCT(ACCT_OTHER));
    global_heap_free(o, sizeof(*o) HEAPACCT(ACCT_OTHER));
    cpu_key(&h);
    run_key = h;
}

static void
cache_path(char *buf, size_t len)
{
    snprintf(buf, len, "%s/%s.pcache", DYNAMO_OPTION(pcache_dir),
             get_application_short_name());
    buf[len - 1] = '\0';
}

static void
open_reader(void)
{
    char path[MAXIMUM_PATH];
    uint64 size64;
    cache_path(path, sizeof(path));
    file_t f = os_open(path, OS_OPEN_READ);
    if (f == INVALID_FILE)
        return;
    if (os_get_file_size_by_handle(f, &size64) && size64 >= sizeof(pc_header_t)) {
        size_t size = (size_t)size64;
        /* Inside DR's reservation: a mapping among the app's would move the app's later
         * mmaps (its libraries) away from where the writer saw them.
         */
        byte *m = os_map_file(f, &size, 0, NULL, MEMPROT_READ,
                              MAP_FILE_COPY_ON_WRITE | MAP_FILE_REACHABLE);
        const pc_header_t *h = (const pc_header_t *)m;
        if (m != NULL && h->magic == PC_MAGIC && h->version == PC_VERSION &&
            h->key == run_key && h->index_off + h->num * sizeof(pc_index_t) <= size &&
            h->data_off + h->data_len <= size) {
            hdr = h;
            index_ = (const pc_index_t *)(m + h->index_off);
            data = m + h->data_off;
        } else if (DYNAMO_OPTION(pcache_stats)) {
            print_file(STDERR, "pcache: %s rejected (%s)\n", path,
                       m == NULL ? "map" : h->key != run_key ? "key" : "header");
        }
    }
    os_close(f);
}

static bool
init(void)
{
    if (!key_done) {
        spin_lock();
        if (!key_done) {
            mode = DYNAMO_OPTION(pcache_dir)[0] == '\0' ? 0 : (int)DYNAMO_OPTION(pcache_mode);
            if (mode != 0)
                compute_key();
            if (mode == 1)
                open_reader();
            key_done = true;
        }
        spin_unlock();
    }
    return mode != 0;
}

bool
pcache_reading(void)
{
    if (mode == 0 || DYNAMO_OPTION(pcache_mode) != 1)
        return false;
    return init() && mode == 1 && hdr != NULL;
}

bool
pcache_writing(void)
{
    if (mode == 0 || DYNAMO_OPTION(pcache_mode) != 2)
        return false;
    return init() && mode == 2;
}

bool
pcache_verifying(void)
{
    return DYNAMO_OPTION(pcache_verify) != 0;
}

/* Module identity memo, keyed by module start. Lookups run under bb_building_lock. */
#define MODMEMO 1024
static struct {
    app_pc start;
    uint64 id;
} modmemo[MODMEMO];

static bool
module_identity(app_pc pc, app_pc *start, uint64 *id)
{
    os_get_module_info_lock();
    module_area_t *ma = module_pc_lookup(pc);
    *start = ma == NULL ? NULL : ma->start;
    os_get_module_info_unlock();
    if (*start == NULL)
        return false;
    uint slot = (uint)(((ptr_uint_t)*start >> 12) % MODMEMO);
    if (modmemo[slot].start != *start) {
        modmemo[slot].id = elf_identity(*start);
        modmemo[slot].start = *start;
    }
    *id = modmemo[slot].id;
    return *id != 0;
}

static bool
unmod_image(app_pc pc)
{
    uint vmf;
    return get_executable_area_vm_flags(pc, &vmf) && TESTANY(PC_VM_UNMOD_IMAGE, vmf) &&
        !TESTANY(PC_VM_WRITABLE, vmf);
}

/*************************************************************************
 * Reader
 */

static const pc_entry_t *
find(app_pc tag)
{
    uint lo = 0, hi = hdr->num;
    while (lo < hi) {
        uint mid = (lo + hi) / 2;
        if (index_[mid].tag < (uint64)tag)
            lo = mid + 1;
        else
            hi = mid;
    }
    if (lo < hdr->num && index_[lo].tag == (uint64)tag)
        return (const pc_entry_t *)(data + index_[lo].off);
    return NULL;
}

#define GET(type, p) (*(const type *)(p))

static instrlist_t *
build_ilist(dcontext_t *dcontext, const pc_entry_t *e, app_pc tag)
{
    const byte *p = (const byte *)(e + 1) + e->src_len, *end = p + e->rec_len;
    instr_t *recs_local[256], **recs = recs_local;
    if (e->nrec > 256)
        recs = (instr_t **)global_heap_alloc(e->nrec * sizeof(instr_t *) HEAPACCT(ACCT_OTHER));
    const byte *icti[256];
    uint nicti = 0;
    bool ok = true;
    uint i;
    for (i = 0; i < e->nrec && p < end && ok; i++) {
        byte kind = *p++;
        instr_t *in = NULL;
        switch (kind) {
        case R_RAW: {
            ushort len = GET(ushort, p);
            in = instr_create(dcontext);
            instr_set_raw_bits(in, (byte *)p + 2, len);
            instr_set_meta(in);
            p += 2 + len;
            break;
        }
        case R_APPREL: {
            ushort off = GET(ushort, p);
            in = instr_create(dcontext);
            instr_set_raw_bits(in, tag + off, p[2]);
            instr_set_rip_rel_pos(in, p[3]);
            instr_set_meta(in);
            p += 4;
            break;
        }
        case R_REL: {
            byte len = p[0];
            app_pc target = (app_pc)GET(uint64, p + 3);
            in = instr_create(dcontext);
            if (decode_from_copy(dcontext, (byte *)p + 11, target - len, in) !=
                p + 11 + len) {
                ok = false;
                break;
            }
            /* re-encoded from its operands at the new place */
            instr_set_raw_bits_valid(in, false);
            instr_set_meta(in);
            p += 11 + len;
            break;
        }
        case R_LABEL:
            in = INSTR_CREATE_label(dcontext);
            break;
        case R_ICTI:
            if (nicti == sizeof(icti) / sizeof(icti[0]))
                ok = false;
            else
                icti[nicti++] = p;
            p += 4;
            break;
        case R_EXIT: {
            ushort opc = GET(ushort, p);
            app_pc target = (app_pc)GET(uint64, p + 11);
            in = opc == OP_jmp ? INSTR_CREATE_jmp(dcontext, opnd_create_pc(target))
                               : INSTR_CREATE_jcc(dcontext, opc, opnd_create_pc(target));
            if (GET(uint, p + 3) != 0)
                instr_set_prefixes(in, GET(uint, p + 3));
            in->flags |= GET(uint, p + 7);
            p += 19;
            break;
        }
        default: ok = false;
        }
        recs[i] = in;
    }
    instrlist_t *ilist = NULL;
    if (ok && i == e->nrec && p == end) {
        ilist = instrlist_create(dcontext);
        uint k = 0;
        for (i = 0; i < e->nrec; i++) {
            if (recs[i] == NULL) {
                const byte *q = icti[k++];
                ushort tgt = GET(ushort, q + 2);
                instr_t *t = tgt < e->nrec ? recs[tgt] : NULL;
                if (t == NULL) {
                    ok = false;
                    break;
                }
                recs[i] = INSTR_CREATE_jcc(dcontext, GET(ushort, q), opnd_create_instr(t));
                instr_set_meta(recs[i]);
            }
            instrlist_append(ilist, recs[i]);
            recs[i] = NULL;
        }
    }
    if (!ok) {
        if (ilist != NULL)
            instrlist_clear_and_destroy(dcontext, ilist);
        ilist = NULL;
    }
    for (i = 0; i < e->nrec; i++) {
        if (recs[i] != NULL && ilist == NULL)
            instr_destroy(dcontext, recs[i]);
    }
    if (recs != recs_local)
        global_heap_free(recs, e->nrec * sizeof(instr_t *) HEAPACCT(ACCT_OTHER));
    return ilist;
}

/* A hit is its entry. Reader and writer state is used under bb_building_lock (shared
 * blocks only); the buffers below rely on that.
 */
static byte srcbuf[MAX_SRC];

static const pcache_hit_t *
lookup(dcontext_t *dcontext, app_pc tag, uint initial_flags, bool avx512_before,
       instrlist_t **ilist, uint *flags, void **vmlist);

const pcache_hit_t *
pcache_lookup(dcontext_t *dcontext, app_pc tag, uint initial_flags, bool avx512_before,
              instrlist_t **ilist, uint *flags, void **vmlist)
{
    uint64 t0 = tsc();
    const pcache_hit_t *h =
        lookup(dcontext, tag, initial_flags, avx512_before, ilist, flags, vmlist);
    t_look += tsc() - t0;
    return h;
}

static const pcache_hit_t *
lookup(dcontext_t *dcontext, app_pc tag, uint initial_flags, bool avx512_before,
       instrlist_t **ilist, uint *flags, void **vmlist)
{
    s_lookup++;
    const pc_entry_t *e = find(tag);
    if (e == NULL) {
        s_noentry++;
        return NULL;
    }
    if (e->initial_flags != initial_flags) {
        s_iflags++;
        return NULL;
    }
    if (e->avx512 != (avx512_before ? 3 : 0)) {
        s_avx++;
        return NULL;
    }
    app_pc mstart;
    uint64 mid;
    if (!module_identity(tag, &mstart, &mid) || mstart != (app_pc)e->mod_start ||
        mid != e->modid) {
        s_module++;
        return NULL;
    }
    /* The vmlist and area flags, as check_new_page_start/contig would make them. */
    app_pc end = tag + e->src_len, stop = NULL;
    void *vml = NULL;
    uint vflags = initial_flags;
    bool ok = check_thread_vm_area(dcontext, tag, tag, &vml, &vflags, &stop, false);
    while (ok && stop < end)
        ok = check_thread_vm_area(dcontext, stop, tag, &vml, &vflags, &stop, true);
    if (!ok || !unmod_image(tag) || !unmod_image(end - 1) ||
        TESTANY(FRAG_SELFMOD_SANDBOXED, vflags) ||
        !TESTALL(vflags & ~initial_flags, e->flags)) {
        s_area++;
        goto miss;
    }
    if (!d_r_safe_read(tag, e->src_len, srcbuf) ||
        memcmp(srcbuf, (const byte *)(e + 1), e->src_len) != 0) {
        s_bytes++;
        goto miss;
    }
    instrlist_t *il = build_ilist(dcontext, e, tag);
    if (il == NULL) {
        s_build++;
        goto miss;
    }
    s_hit++;
    *ilist = il;
    *flags = e->flags;
    *vmlist = vml;
    return (const pcache_hit_t *)e;
miss:
    if (vml != NULL)
        vm_area_destroy_list(dcontext, vml);
    return NULL;
}

static void
dump_bytes(const char *what, const byte *b, uint n)
{
    char line[3 * 96 + 1];
    uint k = 0;
    for (uint i = 0; i < n && i < 96; i++)
        k += snprintf(line + k, sizeof(line) - k, "%02x ", b[i]);
    line[k] = '\0';
    print_file(STDERR, "pcache:   %s %s\n", what, line);
}

static byte vbuf[MAX_SRC * 8], vmask[MAX_SRC * 8];

void
pcache_verify(dcontext_t *dcontext, const pcache_hit_t *hit, fragment_t *f,
              instrlist_t *installed, instrlist_t *fresh, uint fresh_flags)
{
    const pc_entry_t *e = (const pc_entry_t *)hit;
    const char *why = NULL;
    byte *buf = vbuf, *mask = vmask;
    cache_pc entry = FCACHE_ENTRY_PC(f);
    uint len = 0, flen = 0, at = 0;
    instr_t *in;
    for (in = instrlist_first(installed); in != NULL; in = instr_get_next(in))
        len += instr_ok_to_emit(in) ? instr_length(dcontext, in) : 0;
    if (fresh_flags != e->flags)
        why = "flags";
    if (why == NULL && PAD_FRAGMENT_JMPS(f->flags))
        nop_pad_ilist(dcontext, f, fresh, false);
    for (in = instrlist_first(fresh); in != NULL; in = instr_get_next(in)) {
        in->offset = flen;
        flen += instr_ok_to_emit(in) ? instr_length(dcontext, in) : 0;
    }
    if (why == NULL && (flen != len || len > sizeof(vbuf)))
        why = "length";
    if (why == NULL) {
        /* The installed exit ctis, in order, against the fresh ones. */
        instr_t *a = instrlist_first(installed);
        for (in = instrlist_first(fresh); in != NULL && why == NULL;
             in = instr_get_next(in)) {
            if (!instr_is_exit_cti(in))
                continue;
            while (a != NULL && !instr_is_exit_cti(a))
                a = instr_get_next(a);
            linkstub_t *l;
            uint n = 0, k = 0;
            for (instr_t *b = instrlist_first(fresh); b != in; b = instr_get_next(b))
                n += instr_is_exit_cti(b) ? 1 : 0;
            for (l = FRAGMENT_EXIT_STUBS(f); l != NULL && k < n; l = LINKSTUB_NEXT_EXIT(l))
                k++;
            if (a == NULL || l == NULL || instr_get_opcode(a) != instr_get_opcode(in) ||
                (instr_exit_branch_type(a) & ~INSTR_BRANCH_PADDED) !=
                    (instr_exit_branch_type(in) & ~INSTR_BRANCH_PADDED) ||
                (LINKSTUB_NORMAL_DIRECT(l->flags) &&
                 ((direct_linkstub_t *)l)->target_tag != instr_get_branch_target_pc(in)))
                why = "exit";
            if (a != NULL)
                a = instr_get_next(a);
        }
    }
    if (why == NULL) {
        memset(mask, 0, len);
        for (in = instrlist_first(fresh); in != NULL && why == NULL;
             in = instr_get_next(in)) {
            if (!instr_ok_to_emit(in))
                continue;
            uint il = instr_length(dcontext, in);
            if (instr_is_exit_cti(in)) {
                /* Exit displacements point at stubs or linked targets: masked. */
                instr_set_branch_target_pc(in, entry + in->offset);
                memset(mask + in->offset + il - 4, 1, 4);
            }
            if (il > 0 &&
                instr_encode_to_copy(dcontext, in, buf + in->offset, entry + in->offset) ==
                    NULL)
                why = "encode";
        }
        for (at = 0; why == NULL && at < len; at++) {
            if (!mask[at] && buf[at] != entry[at])
                why = "bytes";
        }
        if (why != NULL)
            at--;
    }
    if (why == NULL) {
        s_vok++;
        return;
    }
    s_vbad++;
    if (s_vbad <= 20) {
        print_file(STDERR,
                   "pcache: VERIFY MISMATCH (%s) tag " PFX " at " PFX " flags 0x%x/0x%x "
                   "len %u/%u offset %u\n",
                   why, f->tag, entry, e->flags, fresh_flags, len, flen, at);
        dump_bytes("installed", entry, len);
        if (why[0] == 'b')
            dump_bytes("fresh    ", buf, flen);
        if (s_vbad <= 3) {
            print_file(STDERR, "pcache:   installed list:\n");
            instrlist_disassemble(dcontext, f->tag, installed, STDERR);
            print_file(STDERR, "pcache:   fresh list:\n");
            instrlist_disassemble(dcontext, f->tag, fresh, STDERR);
        }
    }
    if (DYNAMO_OPTION(pcache_verify) == 1)
        os_terminate_with_code(dcontext, TERMINATE_PROCESS, 86);
}

/*************************************************************************
 * Writer
 */

static bool
has_instr_opnd(instr_t *in)
{
    for (int i = 0; i < instr_num_srcs(in); i++) {
        opnd_t o = instr_get_src(in, i);
        if (opnd_is_instr(o) || opnd_is_mem_instr(o))
            return true;
    }
    for (int i = 0; i < instr_num_dsts(in); i++) {
        opnd_t o = instr_get_dst(in, i);
        if (opnd_is_instr(o) || opnd_is_mem_instr(o))
            return true;
    }
    return false;
}

static bool
jmp_or_jcc(int opc, bool short_ok)
{
    if (opc == OP_jmp || (opc >= OP_jo && opc <= OP_jnle))
        return true;
    return short_ok && (opc == OP_jmp_short || (opc >= OP_jo_short && opc <= OP_jnle_short));
}

static void
skip(int reason)
{
    s_skip[reason]++;
}

static pcache_ref_t *
snapshot(dcontext_t *dcontext, app_pc tag, app_pc end_pc, uint initial_flags,
         bool avx512_before, uint flags, instrlist_t *ilist);

pcache_ref_t *
pcache_snapshot(dcontext_t *dcontext, app_pc tag, app_pc end_pc, uint initial_flags,
                bool avx512_before, uint flags, instrlist_t *ilist)
{
    uint64 t0 = tsc();
    pcache_ref_t *r =
        snapshot(dcontext, tag, end_pc, initial_flags, avx512_before, flags, ilist);
    t_snap += tsc() - t0;
    return r;
}

static pcache_ref_t *
snapshot(dcontext_t *dcontext, app_pc tag, app_pc end_pc, uint initial_flags,
         bool avx512_before, uint flags, instrlist_t *ilist)
{
    if (avx512_before != d_r_is_avx512_code_in_use()) {
        skip(0);
        return NULL;
    }
    if (!TESTANY(FRAG_SHARED, flags) ||
        TESTANY(FRAG_SELFMOD_SANDBOXED | FRAG_HAS_SYSCALL | FRAG_HAS_TRANSLATION_INFO |
                    FRAG_COARSE_GRAIN | FRAG_IS_TRACE | FRAG_HAS_DIRECT_CTI | FRAG_32_BIT |
                    FRAG_X86_TO_X64 | FRAG_TEMP_PRIVATE,
                flags) ||
        TESTANY(INSTR_RSEQ_ENDPOINT, ilist->flags) ||
        instrlist_get_return_target(ilist) != NULL ||
        instrlist_get_fall_through_target(ilist) != NULL) {
        skip(1);
        return NULL;
    }
    app_pc mstart;
    uint64 mid;
    if (end_pc <= tag || end_pc - tag > MAX_SRC || !unmod_image(tag) ||
        !unmod_image(end_pc - 1) || !module_identity(tag, &mstart, &mid)) {
        skip(2);
        return NULL;
    }
    uint n = 0;
    instr_t *in;
    for (in = instrlist_first(ilist); in != NULL; in = instr_get_next(in))
        n++;
    if (n > MAX_INSTRS) {
        skip(3);
        return NULL;
    }
    pcache_ref_t *s = (pcache_ref_t *)global_heap_alloc(sizeof(*s) HEAPACCT(ACCT_OTHER));
    memset(s, 0, sizeof(*s));
    s->ins = (struct snap_ins *)global_heap_alloc(n * sizeof(*s->ins) HEAPACCT(ACCT_OTHER));
    memset(s->ins, 0, n * sizeof(*s->ins));
    s->tag = tag;
    s->end = end_pc;
    s->initial_flags = initial_flags;
    s->flags = flags;
    s->avx512 = avx512_before;
    s->n = n;
    bool ok = true;
    uint j = 0;
    for (in = instrlist_first(ilist); in != NULL && ok; in = instr_get_next(in), j++) {
        struct snap_ins *si = &s->ins[j];
        si->in = in;
        si->tgt = -1;
        si->pos = -1;
        if (!instr_ok_to_emit(in)) {
            si->kind = instr_is_label(in) ? 0 : 0xff;
        } else if (instr_is_exit_cti(in)) {
            uint xf = in->flags & EXIT_CTI_TYPES;
            si->kind = R_EXIT;
            si->opcode = (ushort)instr_get_opcode(in);
            si->prefixes = instr_get_prefixes(in);
            si->xflags = xf & ~INSTR_BRANCH_PADDED;
            si->target = instr_get_branch_target_pc(in);
            if (!jmp_or_jcc(si->opcode, false) || instr_is_cti_short_rewrite(in, NULL) ||
                TESTANY(LINK_NI_SYSCALL_ALL | LINK_SPECIAL_EXIT, xf))
                ok = false;
        } else if (instr_operands_valid(in) && instr_is_cti(in) &&
                   opnd_is_instr(instr_get_target(in))) {
            si->kind = R_ICTI;
            si->opcode = (ushort)instr_get_opcode(in);
            instr_t *t = opnd_get_instr(instr_get_target(in)), *x;
            uint k = 0;
            for (x = instrlist_first(ilist); x != NULL && x != t; x = instr_get_next(x))
                k++;
            if (x == NULL || !instr_is_meta(in) || !jmp_or_jcc(si->opcode, true) ||
                instr_get_prefixes(in) != 0)
                ok = false;
            si->tgt = (int)k;
        } else {
            si->kind = R_RAW; /* refined after emit */
            if (instr_operands_valid(in) && has_instr_opnd(in))
                ok = false;
        }
    }
    for (j = 0; j < n && ok; j++) {
        struct snap_ins *si = &s->ins[j];
        if (si->kind == 0xff)
            ok = false;
        else if (si->kind == R_ICTI) {
            struct snap_ins *t = &s->ins[si->tgt];
            if (t->kind == R_ICTI || t->kind == 0xff)
                ok = false;
            t->is_target = true;
        }
    }
    if (!ok) {
        global_heap_free(s->ins, n * sizeof(*s->ins) HEAPACCT(ACCT_OTHER));
        global_heap_free(s, sizeof(*s) HEAPACCT(ACCT_OTHER));
        skip(4);
        return NULL;
    }
    return s;
}

/* A raw instr's emitted bytes: plain (pos -1), or one instruction with a pc-relative field. */
static bool
classify(dcontext_t *dcontext, pcache_ref_t *s, struct snap_ins *si, byte *code, uint len)
{
    instr_t *in = si->in;
    instr_t tmp;
    bool ok = true;
    for (uint o = 0; o < len && ok;) {
        instr_init(dcontext, &tmp);
        byte *next = decode_from_copy(dcontext, code + o, code + o, &tmp);
        app_pc target = NULL;
        int pos = -1;
        if (next == NULL) {
            ok = false;
        } else if (instr_is_cti(&tmp) && opnd_is_pc(instr_get_target(&tmp))) {
            target = opnd_get_pc(instr_get_target(&tmp));
            if (!jmp_or_jcc(instr_get_opcode(&tmp), false) &&
                instr_get_opcode(&tmp) != OP_call)
                ok = false;
            pos = (int)(next - code) - 4;
        } else if (instr_get_rel_addr_target(&tmp, &target)) {
            pos = (int)instr_get_rip_rel_pos(&tmp);
            if (pos <= 0 || pos + 4 > (int)(next - code - o))
                ok = false;
        } else if (instr_get_rel_data_or_instr_target(&tmp, &target)) {
            ok = false; /* some other pc-relative form */
        }
        instr_free(dcontext, &tmp);
        if (ok && pos >= 0) {
            if (o != 0 || next != code + len) /* inside a multi-instruction bundle */
                ok = false;
            si->pos = pos;
            si->target = target;
            if (!instr_is_meta(in) && instr_raw_bits_valid(in) && instr_rip_rel_valid(in) &&
                instr_get_rip_rel_pos(in) == (uint)pos && in->length == len &&
                in->bytes >= s->tag && in->bytes + len <= s->end &&
                memcmp(in->bytes, code, pos) == 0 &&
                memcmp(in->bytes + pos + 4, code + pos + 4, len - pos - 4) == 0)
                si->kind = R_APPREL;
            else
                si->kind = R_REL;
        }
        if (next != NULL)
            o = (uint)(next - code);
    }
    return ok;
}

static byte wbuf[1 << 17];
static cache_pc wpcs[MAX_INSTRS];

static void
record(dcontext_t *dcontext, pcache_ref_t *s, instrlist_t *ilist, fragment_t *f);

void
pcache_record(dcontext_t *dcontext, pcache_ref_t *s, instrlist_t *ilist, fragment_t *f)
{
    uint64 t0 = tsc();
    record(dcontext, s, ilist, f);
    t_rec += tsc() - t0;
}

static void
record(dcontext_t *dcontext, pcache_ref_t *s, instrlist_t *ilist, fragment_t *f)
{
    uint n = s->n, j = 0, nrec = 0;
    cache_pc pc = FCACHE_ENTRY_PC(f);
    instr_t *in;
    bool ok = true;
    spin_lock();
    /* Where each snapshot instr landed; pad nops emit inserted are not in the snapshot. */
    for (in = instrlist_first(ilist); in != NULL; in = instr_get_next(in)) {
        if (j < n && in == s->ins[j].in)
            wpcs[j++] = pc;
        if (instr_ok_to_emit(in))
            pc += instr_length(dcontext, in);
    }
    if (j != n)
        ok = false;
    for (j = 0; j < n && ok; j++) {
        if (s->ins[j].kind == R_RAW)
            ok = classify(dcontext, s, &s->ins[j], wpcs[j],
                          instr_length(dcontext, s->ins[j].in));
    }
    /* Record indices: raw instrs share a bundle unless one is a jump target. */
    bool open = false;
    for (j = 0; j < n && ok; j++) {
        struct snap_ins *si = &s->ins[j];
        si->rec = nrec;
        if (si->kind == 0 && !si->is_target)
            continue;
        if (si->kind == R_RAW && open && !si->is_target)
            continue;
        open = si->kind == R_RAW;
        nrec++;
    }
    pc_entry_t *e = (pc_entry_t *)wbuf;
    byte *p = (byte *)(e + 1), *bundle = NULL;
    memset(e, 0, sizeof(*e));
    e->tag = (uint64)s->tag;
    e->src_len = (ushort)(s->end - s->tag);
    module_identity(s->tag, (app_pc *)&e->mod_start, &e->modid);
    e->initial_flags = s->initial_flags;
    e->flags = s->flags;
    e->avx512 = s->avx512 ? 3 : 0;
    e->nrec = (ushort)nrec;
    memcpy(p, s->tag, e->src_len);
    p += e->src_len;
    byte *rec0 = p;
    for (j = 0; j < n && ok; j++) {
        struct snap_ins *si = &s->ins[j];
        byte cls;
        app_pc t;
        uint len;
        if (p + 64 + MAX_SRC > wbuf + sizeof(wbuf)) {
            ok = false;
            break;
        }
        switch (si->kind) {
        case 0:
            if (si->is_target) {
                bundle = NULL;
                *p++ = R_LABEL;
            }
            break;
        case R_EXIT:
            t = si->target;
            cls = 0;
            bundle = NULL;
            *p++ = R_EXIT;
            *(ushort *)p = si->opcode;
            p[2] = cls;
            *(uint *)(p + 3) = si->prefixes;
            *(uint *)(p + 7) = si->xflags;
            *(uint64 *)(p + 11) = (uint64)t;
            p += 19;
            break;
        case R_ICTI:
            bundle = NULL;
            *p++ = R_ICTI;
            *(ushort *)p = si->opcode;
            *(ushort *)(p + 2) = (ushort)s->ins[si->tgt].rec;
            p += 4;
            break;
        case R_RAW:
            len = instr_length(dcontext, si->in);
            if (bundle == NULL || si->is_target) {
                *p++ = R_RAW;
                bundle = p;
                *(ushort *)p = 0;
                p += 2;
            }
            memcpy(p, wpcs[j], len);
            p += len;
            *(ushort *)bundle += (ushort)len;
            break;
        case R_APPREL:
            bundle = NULL;
            *p++ = R_APPREL;
            *(ushort *)p = (ushort)(si->in->bytes - s->tag);
            p[2] = (byte)si->in->length;
            p[3] = (byte)si->pos;
            p += 4;
            break;
        case R_REL:
            len = instr_length(dcontext, si->in);
            t = si->target;
            cls = 0;
            bundle = NULL;
            *p++ = R_REL;
            p[0] = (byte)len;
            p[1] = (byte)si->pos;
            p[2] = cls;
            *(uint64 *)(p + 3) = (uint64)t;
            memcpy(p + 11, wpcs[j], len);
            memset(p + 11 + si->pos, 0, 4);
            p += 11 + len;
            break;
        }
    }
    e->rec_len = (uint)(p - rec0);
    if (ok) {
        size_t sz = ALIGN_FORWARD((size_t)(p - wbuf), 8);
        if (wnum == wcap) {
            uint ncap = wcap == 0 ? 4096 : wcap * 2;
            byte **n2 =
                (byte **)global_heap_alloc(ncap * sizeof(byte *) HEAPACCT(ACCT_OTHER));
            if (wents != NULL) {
                memcpy(n2, wents, wnum * sizeof(byte *));
                global_heap_free(wents, wcap * sizeof(byte *) HEAPACCT(ACCT_OTHER));
            }
            wents = n2;
            wcap = ncap;
        }
        byte *copy = (byte *)global_heap_alloc(sz + sizeof(size_t) HEAPACCT(ACCT_OTHER));
        *(size_t *)copy = sz;
        memcpy(copy + sizeof(size_t), wbuf, sz);
        wents[wnum++] = copy;
        s_rec++;
    } else
        skip(5);
    spin_unlock();
    global_heap_free(s->ins, n * sizeof(*s->ins) HEAPACCT(ACCT_OTHER));
    global_heap_free(s, sizeof(*s) HEAPACCT(ACCT_OTHER));
}

static uint64
ent_tag(byte *b)
{
    return ((pc_entry_t *)(b + sizeof(size_t)))->tag;
}

/* Heap sort of the entries by tag. */
static void
sift(byte **a, uint i, uint n)
{
    for (;;) {
        uint c = 2 * i + 1;
        if (c >= n)
            return;
        if (c + 1 < n && ent_tag(a[c + 1]) > ent_tag(a[c]))
            c++;
        if (ent_tag(a[i]) >= ent_tag(a[c]))
            return;
        byte *t = a[i];
        a[i] = a[c];
        a[c] = t;
        i = c;
    }
}

/* Buffered through wbuf, free at exit; n == 0 flushes. */
static size_t wfill;

static void
wput(file_t f, const void *b, size_t n, bool *ok)
{
    if (wfill + n > sizeof(wbuf) || n == 0) {
        if (wfill > 0 && os_write(f, wbuf, wfill) != (ssize_t)wfill)
            *ok = false;
        wfill = 0;
    }
    if (n > sizeof(wbuf)) {
        if (os_write(f, b, n) != (ssize_t)n)
            *ok = false;
    } else if (n > 0) {
        memcpy(wbuf + wfill, b, n);
        wfill += n;
    }
}

static void
write_file(void)
{
    char path[MAXIMUM_PATH], tmp[MAXIMUM_PATH];
    uint i, num = 0;
    for (i = wnum / 2; i-- > 0;)
        sift(wents, i, wnum);
    for (i = wnum; i-- > 1;) {
        byte *t = wents[0];
        wents[0] = wents[i];
        wents[i] = t;
        sift(wents, 0, i);
    }
    /* drop later builds of a tag (rebuilt after a flush) */
    for (i = 0; i < wnum; i++) {
        if (num == 0 || ent_tag(wents[i]) != ent_tag(wents[num - 1]))
            wents[num++] = wents[i];
    }
    cache_path(path, sizeof(path));
    snprintf(tmp, sizeof(tmp), "%s.%d.tmp", path, get_process_id());
    tmp[sizeof(tmp) - 1] = '\0';
    file_t f = os_open(tmp, OS_OPEN_WRITE | OS_OPEN_REQUIRE_NEW);
    if (f == INVALID_FILE) {
        print_file(STDERR, "pcache: cannot write %s\n", tmp);
        return;
    }
    pc_header_t h;
    memset(&h, 0, sizeof(h));
    h.magic = PC_MAGIC;
    h.version = PC_VERSION;
    h.num = num;
    h.key = run_key;
    h.vm_lo = (uint64)vm_lo;
    h.vm_hi = (uint64)vm_hi;
    h.index_off = sizeof(h);
    h.data_off = h.index_off + ALIGN_FORWARD(num * sizeof(pc_index_t), 8);
    uint64 off = 0;
    for (i = 0; i < num; i++)
        off += *(size_t *)wents[i];
    h.data_len = off;
    bool ok = true;
    wfill = 0;
    wput(f, &h, sizeof(h), &ok);
    off = 0;
    for (i = 0; i < num; i++) {
        pc_index_t x = { ent_tag(wents[i]), off };
        wput(f, &x, sizeof(x), &ok);
        off += *(size_t *)wents[i];
    }
    for (i = 0; i < num; i++)
        wput(f, wents[i] + sizeof(size_t), *(size_t *)wents[i], &ok);
    wput(f, NULL, 0, &ok);
    os_close(f);
    if (ok && os_rename_file(tmp, path, true)) {
        if (DYNAMO_OPTION(pcache_stats))
            print_file(STDERR, "pcache: wrote %u entries, %llu bytes, to %s\n", num,
                       (unsigned long long)(h.data_off + h.data_len), path);
    } else {
        print_file(STDERR, "pcache: writing %s failed\n", path);
        os_delete_file(tmp);
    }
}

void
pcache_process_exit(void)
{
    if (mode <= 0 || written)
        return;
    written = true;
    if (mode == 2) {
        /* Bounded: a thread stopped inside the lock at exit must not hang the process. */
        int tries = 0;
        while (!atomic_compare_exchange_int(&wlock, 0, 1)) {
            if (++tries > 1000) {
                print_file(STDERR, "pcache: writer busy at exit, not written\n");
                return;
            }
            os_thread_yield();
        }
        uint64 t0 = tsc();
        write_file();
        t_write += tsc() - t0;
        spin_unlock();
    }
    if (DYNAMO_OPTION(pcache_stats)) {
        print_file(STDERR,
                   "pcache: pid %d mode %d lookups %u hits %u noentry %u iflags %u avx %u "
                   "module %u area %u bytes %u build %u verify_ok %u verify_bad %u "
                   "recorded %u skip %u/%u/%u/%u/%u/%u Mcycles snap %llu rec %llu write %llu "
                   "lookup %llu\n",
                   get_process_id(), mode, s_lookup, s_hit, s_noentry, s_iflags, s_avx,
                   s_module, s_area, s_bytes, s_build, s_vok, s_vbad, s_rec, s_skip[0],
                   s_skip[1], s_skip[2], s_skip[3], s_skip[4], s_skip[5], t_snap >> 20,
                   t_rec >> 20, t_write >> 20, t_look >> 20);
    }
}
