/* Translation cache across runs: see pcache.h and lhns/v3compat docs/translation-cache.md.
 *
 * An entry is a finished basic block (after the client and mangling) recorded as a short
 * instruction list: raw byte runs, app instructions with a rip-relative field (replayed from the
 * app bytes, as DR builds them), DR/client instructions with a pc-relative field, intra-block
 * jumps by index, and symbolic exit ctis. A hit replays that list through emit_fragment_ex(), so
 * padding, stubs, linkstubs, vmarea registration, linking and the fragment table are DR's
 * ordinary code at the real address.
 *
 * Nothing stored is absolute. Every address is a value record: an offset from the block's own
 * module, DR, or a private module, or a named IBL or clean-call routine. Each value also stores
 * the layout predicates DR's choices depend on (rel32 reach from vmcode, 32-bit fit); the
 * reader's must be equal, or the block is a miss.
 *
 * Absolute immediates in raw runs are found by decoding the emitted code, not where DR creates
 * them, because a client's clean-call arguments are opaque to DR. An immediate is relocated only
 * when it equals an address derived from the block's own bytes (an instruction's pc, its end, a
 * rip-relative or branch target); any other immediate that lands in mapped memory makes the
 * block uncacheable.
 *
 * A hit needs: the file's key (DR and private module builds, TLS layout, options, CPUID, XCR0);
 * the same initial flags and AVX-512 state; the module's identity, size and offset; the range in
 * never-written image areas (VM_UNMOD_IMAGE); the app bytes equal to the recorded ones; every
 * value's predicates equal. Anything else, including any malformed byte, is a miss.
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
#include <stddef.h>

#define PC_MAGIC 0x3265686361637076ULL /* "vpcache2" */
#define PC_VERSION 2
#define MAX_SRC 4096
#define MAX_INSTRS 2048
#define MAX_DERIVED 1024
#define MAX_FIX 64
#define MAX_PRIV 32
#define KEY_MAX 8192
/* vmareas.c's private VM_WRITABLE and VM_UNMOD_IMAGE */
#define PC_VM_WRITABLE 0x1
#define PC_VM_UNMOD_IMAGE 0x2

/* Records. */
enum { R_RAW = 1, R_APPREL, R_REL, R_LABEL, R_ICTI, R_EXIT };
/* Value classes. */
enum { V_MOD = 1, V_DR, V_PRIV, V_IBL, V_CCSAVE, V_CCRESTORE };
/* Layout predicates of a value. */
#define PRE_REACH 0x1
#define PRE_S32 0x2
#define PRE_U32 0x4
/* Entry flags. */
#define E_PIN 0x1 /* a 32-bit immediate holds a module address: same base only */

/* On-disk layout; tools/pcache.py in v3compat reads and writes the same. Little-endian. */
typedef struct {
    uint64 magic;
    uint version;
    uint header_size;
    uint64 file_size;
    uint64 header_hash; /* of this header with header_hash 0 */
    uint64 body_hash;   /* of [header_size, file_size) */
    uint key_len;
    uint num;
    uint64 key_off, index_off, data_off, data_len;
} pc_header_t;

typedef struct {
    uint64 modid;
    uint off; /* of the tag in its module */
    uint len; /* of the entry */
    uint64 data_off;
} pc_index_t;

typedef struct {
    uint64 hash; /* of the entry's bytes after this field */
    uint64 modid;
    uint mod_off;
    uint mod_size;
    uint64 mod_start; /* checked only with E_PIN */
    uint initial_flags;
    uint flags;
    ushort src_len;
    ushort nrec;
    byte avx512;
    byte eflags;
    ushort pad;
    uint rec_len;
    uint pad2;
    /* then src bytes, then records */
} pc_entry_t;

typedef struct {
    byte cls;
    byte pre;
    ushort arg;
    uint pad;
    uint64 off;
} pc_val_t;

/* Record sizes after the kind byte. */
#define SZ_RAW_HDR 4 /* len, nfix */
#define SZ_FIX (4 + sizeof(pc_val_t))
#define SZ_APPREL 4
#define SZ_REL_HDR (2 + sizeof(pc_val_t))
#define SZ_ICTI 4
#define SZ_EXIT (10 + sizeof(pc_val_t))

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
static char key[KEY_MAX];
static uint key_len;
static struct {
    app_pc base;
    size_t size;
} priv[MAX_PRIV];
static uint npriv;

/* reader */
static const pc_header_t *hdr;
static const pc_index_t *index_;
static const byte *data;

/* writer: entries kept in DR heap until exit */
static volatile int wlock;
static byte **wents;
static uint wnum, wcap;
static size_t wbytes;
static bool written;

/* stats */
static uint s_lookup, s_hit, s_noentry, s_corrupt, s_iflags, s_avx, s_module, s_pin, s_area,
    s_bytes, s_build, s_pre;
static uint s_vok, s_vbad, s_rec, s_cap, s_skip[6];
static uint s_bad_kind[8]; /* build failures by the record being replayed */
/* why the writer could not make a block position-independent */
enum { F_IMM_DR, F_IMM_PRIV, F_IMM_VMM, F_IMM_MOD, F_IMM_MAPPED, F_REL, F_EXIT, F_FIX, F_NUM };
static uint s_fail[F_NUM];

/* XXH64 (seed 0): a checksum against corruption, not an integrity guarantee. */
#define XP1 11400714785074694791ULL
#define XP2 14029467366897019727ULL
#define XP3 1609587929392839161ULL
#define XP4 9650029242287828579ULL
#define XP5 2870177450012600261ULL

static inline uint64
rotl64(uint64 x, int r)
{
    return (x << r) | (x >> (64 - r));
}

static inline uint64
rd64(const byte *p)
{
    uint64 v;
    memcpy(&v, p, sizeof(v));
    return v;
}

static inline uint
rd32(const byte *p)
{
    uint v;
    memcpy(&v, p, sizeof(v));
    return v;
}

static inline ushort
rd16(const byte *p)
{
    ushort v;
    memcpy(&v, p, sizeof(v));
    return v;
}

static inline uint64
xround(uint64 acc, uint64 in)
{
    acc += in * XP2;
    acc = rotl64(acc, 31);
    return acc * XP1;
}

static inline uint64
xmerge(uint64 acc, uint64 v)
{
    acc ^= xround(0, v);
    return acc * XP1 + XP4;
}

static uint64
hash64(const void *buf, size_t len)
{
    const byte *p = (const byte *)buf, *end = p + len;
    uint64 h;
    if (len >= 32) {
        uint64 v1 = XP1 + XP2, v2 = XP2, v3 = 0, v4 = 0 - XP1;
        do {
            v1 = xround(v1, rd64(p));
            v2 = xround(v2, rd64(p + 8));
            v3 = xround(v3, rd64(p + 16));
            v4 = xround(v4, rd64(p + 24));
            p += 32;
        } while (p + 32 <= end);
        h = rotl64(v1, 1) + rotl64(v2, 7) + rotl64(v3, 12) + rotl64(v4, 18);
        h = xmerge(h, v1);
        h = xmerge(h, v2);
        h = xmerge(h, v3);
        h = xmerge(h, v4);
    } else
        h = XP5;
    h += len;
    while (p + 8 <= end) {
        h ^= xround(0, rd64(p));
        h = rotl64(h, 27) * XP1 + XP4;
        p += 8;
    }
    if (p + 4 <= end) {
        h ^= (uint64)rd32(p) * XP1;
        h = rotl64(h, 23) * XP2 + XP3;
        p += 4;
    }
    while (p < end) {
        h ^= (*p) * XP5;
        h = rotl64(h, 11) * XP1;
        p++;
    }
    h ^= h >> 33;
    h *= XP2;
    h ^= h >> 29;
    h *= XP3;
    h ^= h >> 32;
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
                return hash64(note + desc_off, nh->n_descsz) | 1;
            o = next;
        }
    }
    uint64 h[2] = { hash64(&eh, sizeof(eh)), hash64(ph, eh.e_phnum * sizeof(Elf64_Phdr)) };
    return hash64(h, sizeof(h)) & ~1ULL;
}

static void
key_add(const char *fmt, ...)
{
    va_list ap;
    if (key_len >= sizeof(key) - 1)
        return;
    va_start(ap, fmt);
    int n = vsnprintf(key + key_len, sizeof(key) - key_len, fmt, ap);
    va_end(ap);
    if (n < 0 || key_len + n >= sizeof(key) - 1)
        key_len = sizeof(key); /* overflow: no key, the cache stays off */
    else
        key_len += n;
}

/* The full key, compared byte for byte with the file's. No base address is in it: entries are
 * position-independent.
 */
static void
compute_key(void)
{
    uint r[4];
    key_len = 0;
    key_add("v3pcache %d\n", PC_VERSION);
    key_add("dr %llx\n", elf_identity(get_dynamorio_dll_start()));
    acquire_recursive_lock(&privload_lock);
    for (privmod_t *m = privload_first_module(); m != NULL; m = privload_next_module(m)) {
        if (m->externally_loaded)
            continue;
        if (npriv == MAX_PRIV) {
            key_len = sizeof(key);
            break;
        }
        priv[npriv].base = m->base;
        priv[npriv].size = m->size;
        npriv++;
        key_add("priv %s %llx %llx\n", m->name, elf_identity(m->base), (uint64)m->size);
    }
    release_recursive_lock(&privload_lock);
    /* DR's TLS layout is part of its build; the client's raw TLS slots depend on its init. */
    uint tls_first;
    uint64 tls_used;
    os_tls_client_layout(&tls_first, &tls_used);
    key_add("tls %x %llx\n", tls_first, tls_used);
    /* Every option, the client's included, but the cache's own. */
    options_t *o = (options_t *)global_heap_alloc(sizeof(*o) HEAPACCT(ACCT_OTHER));
    char *s = (char *)global_heap_alloc(MAX_OPTIONS_STRING HEAPACCT(ACCT_OTHER));
    *o = dynamo_options;
    o->pcache_mode = 0;
    o->pcache_verify = 0;
    o->pcache_stats = 0;
    o->pcache_max_size = 0;
    o->pcache_dir[0] = '\0';
    get_dynamo_options_string(o, s, MAX_OPTIONS_STRING, true);
    key_add("opts %s\n", s);
    global_heap_free(s, MAX_OPTIONS_STRING HEAPACCT(ACCT_OTHER));
    global_heap_free(o, sizeof(*o) HEAPACCT(ACCT_OTHER));
    /* DR's clean-call save and restore depend on the SIMD state (design, "Key"). */
    static const uint leaves[][2] = { { 1, 0 }, { 7, 0 }, { 7, 1 }, { 0xd, 0 }, { 0x80000001, 0 } };
    key_add("cpu");
    for (uint i = 0; i < sizeof(leaves) / sizeof(leaves[0]); i++) {
        __asm__ __volatile__("cpuid"
                             : "=a"(r[0]), "=b"(r[1]), "=c"(r[2]), "=d"(r[3])
                             : "a"(leaves[i][0]), "c"(leaves[i][1]));
        if (leaves[i][0] == 1)
            r[1] = 0; /* APIC id, CLFLUSH size, logical count */
        key_add(" %x.%x.%x.%x", r[0], r[1], r[2], r[3]);
    }
    uint lo, hi;
    __asm__ __volatile__("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
    key_add(" xcr0 %x.%x\n", hi, lo);
    if (DYNAMO_OPTION(pcache_stats) > 1 && key_len < sizeof(key)) {
        byte *rs, *re;
        vmcode_get_reachable_region(&rs, &re);
        print_file(STDERR, "pcache: key\n%spcache: layout dr " PFX " vmcode " PFX "-" PFX
                   " reach " PFX "-" PFX "\n", key, get_dynamorio_dll_start(),
                   vmcode_get_start(), vmcode_get_end(), rs, re);
    }
}

static void
cache_path(char *buf, size_t len)
{
    snprintf(buf, len, "%s/%s.pcache", DYNAMO_OPTION(pcache_dir),
             get_application_short_name());
    buf[len - 1] = '\0';
}

/* The file's structure, every offset bounded; NULL if it holds. */
static const char *
check_file(const byte *m, size_t size)
{
    const pc_header_t *h = (const pc_header_t *)m;
    if (size < sizeof(*h) || h->magic != PC_MAGIC)
        return "magic";
    if (h->version != PC_VERSION || h->header_size != sizeof(*h))
        return "version";
    if (h->file_size != size)
        return "size";
    pc_header_t c = *h;
    c.header_hash = 0;
    if (hash64(&c, sizeof(c)) != h->header_hash)
        return "header hash";
    if (hash64(m + sizeof(*h), size - sizeof(*h)) != h->body_hash)
        return "body hash";
    if (h->key_off > size || h->key_len > size - h->key_off || h->index_off > size ||
        h->num > (size - h->index_off) / sizeof(pc_index_t) || h->data_off > size ||
        h->data_len > size - h->data_off || !ALIGNED(h->index_off, 8))
        return "layout";
    if (h->key_len != key_len || memcmp(m + h->key_off, key, key_len) != 0)
        return "key";
    return NULL;
}

static void
open_reader(void)
{
    char path[MAXIMUM_PATH];
    uint64 size64;
    const char *why = NULL;
    cache_path(path, sizeof(path));
    file_t f = os_open(path, OS_OPEN_READ);
    if (f == INVALID_FILE)
        return;
    if (!os_get_file_size_by_handle(f, &size64) || size64 < sizeof(pc_header_t))
        why = "size";
    else if (size64 > DYNAMO_OPTION(pcache_max_size))
        why = "too large";
    else {
        size_t size = (size_t)size64;
        /* Anywhere: entries do not depend on where the file or the app's mappings are. */
        byte *m = os_map_file(f, &size, 0, NULL, MEMPROT_READ, MAP_FILE_COPY_ON_WRITE);
        if (m == NULL || size != (size_t)size64)
            why = "map";
        else
            why = check_file(m, size);
        if (why == NULL) {
            hdr = (const pc_header_t *)m;
            index_ = (const pc_index_t *)(m + hdr->index_off);
            data = m + hdr->data_off;
            if (DYNAMO_OPTION(pcache_stats)) {
                print_file(STDERR, "pcache: %s opened, %u entries, %llu bytes\n", path,
                           hdr->num, (unsigned long long)size64);
            }
        } else if (m != NULL)
            os_unmap_file(m, size);
    }
    if (why != NULL && DYNAMO_OPTION(pcache_stats))
        print_file(STDERR, "pcache: %s rejected (%s)\n", path, why);
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
            if (key_len >= sizeof(key))
                mode = 0;
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

/* Module identity memo, keyed by module start, cleared at unload. */
#define MODMEMO 1024
static struct {
    app_pc start, end;
    uint64 id;
} modmemo[MODMEMO];

static bool
module_identity(app_pc pc, app_pc *start, app_pc *end, uint64 *id)
{
    os_get_module_info_lock();
    module_area_t *ma = module_pc_lookup(pc);
    *start = ma == NULL ? NULL : ma->start;
    *end = ma == NULL ? NULL : ma->end;
    os_get_module_info_unlock();
    if (*start == NULL)
        return false;
    uint slot = (uint)(((ptr_uint_t)*start >> 12) % MODMEMO);
    spin_lock();
    bool known = modmemo[slot].start == *start && modmemo[slot].end == *end;
    *id = modmemo[slot].id;
    spin_unlock();
    if (!known) {
        *id = elf_identity(*start);
        spin_lock();
        modmemo[slot].start = *start;
        modmemo[slot].end = *end;
        modmemo[slot].id = *id;
        spin_unlock();
    }
    return *id != 0;
}

void
pcache_module_unload(app_pc base)
{
    if (mode <= 0)
        return;
    uint slot = (uint)(((ptr_uint_t)base >> 12) % MODMEMO);
    spin_lock();
    if (modmemo[slot].start == base)
        modmemo[slot].start = NULL;
    spin_unlock();
}

static bool
unmod_image(app_pc pc)
{
    uint vmf;
    return get_executable_area_vm_flags(pc, &vmf) && TESTANY(PC_VM_UNMOD_IMAGE, vmf) &&
        !TESTANY(PC_VM_WRITABLE, vmf);
}

/* The reach mangle_rel_addr() decides by. */
static bool
reaches(app_pc v)
{
    return DYNAMO_OPTION(stable_reach) ? rel32_reachable_from_any_vmcode_offset(v)
                                       : rel32_reachable_from_vmcode(v);
}

static byte
predicates(app_pc v)
{
    ptr_int_t s = (ptr_int_t)v;
    return (reaches(v) ? PRE_REACH : 0) |
        (s >= INT_MIN && s <= INT_MAX ? PRE_S32 : 0) |
        ((ptr_uint_t)v <= UINT_MAX ? PRE_U32 : 0);
}

/*************************************************************************
 * Reader
 */

/* The value's address in this process, if its predicates hold here. */
static bool
resolve(dcontext_t *dcontext, const byte *p, app_pc mstart, size_t msize, app_pc *out)
{
    pc_val_t v;
    memcpy(&v, p, sizeof(v));
    app_pc a = NULL;
    switch (v.cls) {
    case V_MOD:
        if (v.off > msize)
            return false;
        a = mstart + v.off;
        break;
    case V_DR:
        if (v.off >= (uint64)(get_dynamorio_dll_end() - get_dynamorio_dll_start()))
            return false;
        a = get_dynamorio_dll_start() + v.off;
        break;
    case V_PRIV:
        if (v.arg >= npriv || v.off >= priv[v.arg].size)
            return false;
        a = priv[v.arg].base + v.off;
        break;
    case V_IBL:
        if ((v.arg & 0xf) >= IBL_LINK_STATE_END || ((v.arg >> 4) & 0xf) >= IBL_SOURCE_TYPE_END ||
            (v.arg >> 8) >= IBL_BRANCH_TYPE_END || v.off > GENCODE_FROM_DCONTEXT)
            return false;
        a = get_ibl_routine_ex(dcontext, (ibl_entry_point_type_t)(v.arg & 0xf),
                               (ibl_source_fragment_type_t)((v.arg >> 4) & 0xf),
                               (ibl_branch_type_t)(v.arg >> 8),
                               (gencode_mode_t)v.off);
        break;
    case V_CCSAVE:
    case V_CCRESTORE:
        if (v.off > GENCODE_FROM_DCONTEXT)
            return false;
        a = v.cls == V_CCSAVE ? get_clean_call_save(dcontext, (gencode_mode_t)v.off)
                              : get_clean_call_restore(dcontext, (gencode_mode_t)v.off);
        break;
    default: return false;
    }
    if (a == NULL)
        return false;
    if (predicates(a) != v.pre) {
        s_pre++;
        if (DYNAMO_OPTION(pcache_stats) > 1 && s_pre <= 10) {
            byte *rs, *re;
            vmcode_get_reachable_region(&rs, &re);
            print_file(STDERR,
                       "pcache: predicate class %d off 0x%llx at " PFX " was 0x%x now 0x%x; "
                       "module " PFX " dr " PFX " vmcode " PFX "-" PFX " reach " PFX "-" PFX
                       "\n",
                       v.cls, v.off, a, v.pre, predicates(a), mstart,
                       get_dynamorio_dll_start(), vmcode_get_start(), vmcode_get_end(), rs, re);
        }
        return false;
    }
    *out = a;
    return true;
}

/* The entry for (modid, off), or NULL; bounds of the index and entry checked. */
static const pc_entry_t *
find(uint64 modid, uint off, uint *len)
{
    uint lo = 0, hi = hdr->num;
    while (lo < hi) {
        uint mid = (lo + hi) / 2;
        if (index_[mid].modid < modid ||
            (index_[mid].modid == modid && index_[mid].off < off))
            lo = mid + 1;
        else
            hi = mid;
    }
    if (lo >= hdr->num || index_[lo].modid != modid || index_[lo].off != off)
        return NULL;
    const pc_index_t *x = &index_[lo];
    if (x->data_off > hdr->data_len || x->len > hdr->data_len - x->data_off ||
        x->len < sizeof(pc_entry_t) || !ALIGNED(x->data_off, 8)) {
        s_corrupt++;
        return NULL;
    }
    const pc_entry_t *e = (const pc_entry_t *)(data + x->data_off);
    if (hash64((const byte *)e + sizeof(e->hash), x->len - sizeof(e->hash)) != e->hash ||
        e->modid != modid || e->mod_off != off ||
        sizeof(*e) + (size_t)e->src_len + e->rec_len != x->len || e->src_len == 0) {
        s_corrupt++;
        return NULL;
    }
    *len = x->len;
    return e;
}

/* Bounds-checked reads of the record stream. */
#define NEED(n)                      \
    do {                             \
        if ((size_t)(end - p) < (n)) \
            goto bad;                \
    } while (0)

static instrlist_t *
build_ilist(dcontext_t *dcontext, const pc_entry_t *e, app_pc tag, app_pc mstart,
            size_t msize)
{
    const byte *p = (const byte *)(e + 1) + e->src_len, *end = p + e->rec_len;
    instr_t *recs_local[256], **recs = recs_local;
    const byte *icti_local[256], **icti = icti_local;
    uint i = 0, nicti = 0;
    instrlist_t *ilist = NULL;
    if (e->nrec == 0)
        return NULL;
    if (e->nrec > 256) {
        recs = (instr_t **)global_heap_alloc(e->nrec * sizeof(instr_t *) HEAPACCT(ACCT_OTHER));
        icti = (const byte **)global_heap_alloc(e->nrec * sizeof(byte *) HEAPACCT(ACCT_OTHER));
    }
    memset(recs, 0, e->nrec * sizeof(instr_t *));
    byte kind = 0;
    for (i = 0; i < e->nrec; i++) {
        NEED(1);
        kind = *p++;
        instr_t *in = NULL;
        app_pc t;
        switch (kind) {
        case R_RAW: {
            NEED(SZ_RAW_HDR);
            ushort len = rd16(p), nfix = rd16(p + 2);
            p += SZ_RAW_HDR;
            NEED((size_t)len + (size_t)nfix * SZ_FIX);
            if (len == 0)
                goto bad;
            in = instr_create(dcontext);
            recs[i] = in;
            if (nfix == 0)
                instr_set_raw_bits(in, (byte *)p, len);
            else {
                instr_allocate_raw_bits(dcontext, in, len);
                memcpy(in->bytes, p, len);
                const byte *fx = p + len;
                for (uint k = 0; k < nfix; k++, fx += SZ_FIX) {
                    ushort pos = rd16(fx), pos_hi = rd16(fx + 2);
                    if (!resolve(dcontext, fx + 4, mstart, msize, &t))
                        goto bad;
                    uint64 v = (uint64)t;
                    if (pos_hi == 0xffff) {
                        if ((uint)pos + 8 > len)
                            goto bad;
                        memcpy(in->bytes + pos, &v, 8);
                    } else {
                        uint lo32 = (uint)v, hi32 = (uint)(v >> 32);
                        if ((uint)pos + 4 > len || (uint)pos_hi + 4 > len)
                            goto bad;
                        memcpy(in->bytes + pos, &lo32, 4);
                        memcpy(in->bytes + pos_hi, &hi32, 4);
                    }
                }
            }
            instr_set_meta(in);
            p += len + (size_t)nfix * SZ_FIX;
            break;
        }
        case R_APPREL: {
            NEED(SZ_APPREL);
            ushort off = rd16(p);
            byte len = p[2], pos = p[3];
            p += SZ_APPREL;
            if (len == 0 || pos == 0 || (uint)pos + 4 > len || (uint)off + len > e->src_len)
                goto bad;
            /* DR left it rip-relative because it reached from vmcode: it must here too. */
            t = tag + off + len + (int)rd32((const byte *)(e + 1) + off + pos);
            if (!reaches(t)) {
                s_pre++;
                goto bad;
            }
            in = instr_create(dcontext);
            recs[i] = in;
            instr_set_raw_bits(in, tag + off, len);
            instr_set_rip_rel_pos(in, pos);
            instr_set_meta(in);
            break;
        }
        case R_REL: {
            NEED(SZ_REL_HDR);
            byte len = p[0], pos = p[1];
            if (len == 0 || len > 17 || (uint)pos + 4 > len)
                goto bad;
            NEED(SZ_REL_HDR + (size_t)len);
            if (!resolve(dcontext, p + 2, mstart, msize, &t))
                goto bad;
            if (!rel32_reachable_from_vmcode(t)) {
                s_pre++;
                goto bad;
            }
            /* zero-padded: the decoder must not read past the record */
            byte buf[32];
            memset(buf, 0, sizeof(buf));
            memcpy(buf, p + SZ_REL_HDR, len);
            in = instr_create(dcontext);
            recs[i] = in;
            /* the field is zero, so decoding at target - len yields target */
            if (decode_from_copy(dcontext, buf, t - len, in) != buf + len)
                goto bad;
            /* re-encoded from its operands at the new place */
            instr_set_raw_bits_valid(in, false);
            instr_set_meta(in);
            p += SZ_REL_HDR + len;
            break;
        }
        case R_LABEL:
            in = INSTR_CREATE_label(dcontext);
            recs[i] = in;
            break;
        case R_ICTI:
            NEED(SZ_ICTI);
            icti[nicti++] = p;
            p += SZ_ICTI;
            break;
        case R_EXIT: {
            NEED(SZ_EXIT);
            ushort opc = rd16(p);
            if (opc != OP_jmp && (opc < OP_jo || opc > OP_jnle))
                goto bad;
            if (!resolve(dcontext, p + 10, mstart, msize, &t))
                goto bad;
            in = opc == OP_jmp ? INSTR_CREATE_jmp(dcontext, opnd_create_pc(t))
                               : INSTR_CREATE_jcc(dcontext, opc, opnd_create_pc(t));
            recs[i] = in;
            if (rd32(p + 2) != 0)
                instr_set_prefixes(in, rd32(p + 2));
            in->flags |= rd32(p + 6) & EXIT_CTI_TYPES;
            p += SZ_EXIT;
            break;
        }
        default: goto bad;
        }
    }
    if (p != end)
        goto bad;
    /* intra-block jumps, now that their targets exist */
    uint k = 0;
    for (i = 0; i < e->nrec; i++) {
        if (recs[i] != NULL)
            continue;
        const byte *q = icti[k++];
        ushort opc = rd16(q), tgt = rd16(q + 2);
        bool short_ok = opc == OP_jmp_short || (opc >= OP_jo_short && opc <= OP_jnle_short);
        if (tgt >= e->nrec || recs[tgt] == NULL || tgt == i ||
            !(opc == OP_jmp || (opc >= OP_jo && opc <= OP_jnle) || short_ok))
            goto bad;
        /* an ICTI's target is never another ICTI: recs[tgt] was built above */
        recs[i] = INSTR_CREATE_jcc(dcontext, opc, opnd_create_instr(recs[tgt]));
        instr_set_meta(recs[i]);
    }
    ilist = instrlist_create(dcontext);
    for (i = 0; i < e->nrec; i++) {
        instrlist_append(ilist, recs[i]);
        recs[i] = NULL;
    }
    goto done;
bad:
    s_bad_kind[kind < 8 ? kind : 0]++;
    for (i = 0; i < e->nrec; i++) {
        if (recs[i] != NULL)
            instr_destroy(dcontext, recs[i]);
    }
done:
    if (recs != recs_local) {
        global_heap_free(recs, e->nrec * sizeof(instr_t *) HEAPACCT(ACCT_OTHER));
        global_heap_free(icti, e->nrec * sizeof(byte *) HEAPACCT(ACCT_OTHER));
    }
    return ilist;
}

/* A hit is its entry. Reader and writer state is used under bb_building_lock (shared
 * blocks only); the buffers below rely on that.
 */
static byte srcbuf[MAX_SRC];
static app_pc vtargets[MAX_INSTRS];
static uint nvtargets;

const pcache_hit_t *
pcache_lookup(dcontext_t *dcontext, app_pc tag, uint initial_flags, bool avx512_before,
              instrlist_t **ilist, uint *flags, void **vmlist)
{
    s_lookup++;
    app_pc mstart, mend;
    uint64 mid;
    uint elen;
    if (!module_identity(tag, &mstart, &mend, &mid) || mend - mstart > UINT_MAX) {
        s_noentry++;
        return NULL;
    }
    const pc_entry_t *e = find(mid, (uint)(tag - mstart), &elen);
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
    if (e->mod_size != (uint)(mend - mstart) || tag + e->src_len > mend) {
        s_module++;
        return NULL;
    }
    if (TESTANY(E_PIN, e->eflags) && (app_pc)e->mod_start != mstart) {
        s_pin++;
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
    instrlist_t *il = build_ilist(dcontext, e, tag, mstart, mend - mstart);
    if (il == NULL) {
        s_build++;
        goto miss;
    }
    if (pcache_verifying()) {
        /* emit rewrites exit targets: verify compares the ones built here */
        nvtargets = 0;
        for (instr_t *in = instrlist_first(il); in != NULL; in = instr_get_next(in)) {
            if (instr_is_exit_cti(in) && nvtargets < MAX_INSTRS)
                vtargets[nvtargets++] = instr_get_branch_target_pc(in);
        }
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
        /* Every exit, in order: type and target (relocated app pc or rebuilt IBL routine,
         * as built before emit rewrote it), before the masked byte compare hides it.
         */
        instr_t *a = instrlist_first(installed);
        uint n = 0;
        for (in = instrlist_first(fresh); in != NULL && why == NULL;
             in = instr_get_next(in)) {
            if (!instr_is_exit_cti(in))
                continue;
            while (a != NULL && !instr_is_exit_cti(a))
                a = instr_get_next(a);
            if (a == NULL || n >= nvtargets || instr_get_opcode(a) != instr_get_opcode(in) ||
                (instr_exit_branch_type(a) & ~INSTR_BRANCH_PADDED) !=
                    (instr_exit_branch_type(in) & ~INSTR_BRANCH_PADDED) ||
                vtargets[n] != instr_get_branch_target_pc(in))
                why = "exit";
            if (a != NULL)
                a = instr_get_next(a);
            n++;
        }
        if (why == NULL && n != nvtargets)
            why = "exit";
        /* and the stubs DR made from them */
        linkstub_t *l = FRAGMENT_EXIT_STUBS(f);
        for (in = instrlist_first(fresh); in != NULL && why == NULL;
             in = instr_get_next(in)) {
            if (!instr_is_exit_cti(in))
                continue;
            if (l == NULL ||
                (LINKSTUB_NORMAL_DIRECT(l->flags) &&
                 ((direct_linkstub_t *)l)->target_tag != instr_get_branch_target_pc(in)))
                why = "stub";
            if (l != NULL)
                l = LINKSTUB_NEXT_EXIT(l);
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

pcache_ref_t *
pcache_snapshot(dcontext_t *dcontext, app_pc tag, app_pc end_pc, uint initial_flags,
                bool avx512_before, uint flags, instrlist_t *ilist)
{
    if (avx512_before != d_r_is_avx512_code_in_use()) {
        s_skip[0]++;
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
        s_skip[1]++;
        return NULL;
    }
    app_pc mstart, mend;
    uint64 mid;
    if (end_pc <= tag || end_pc - tag > MAX_SRC || !unmod_image(tag) ||
        !unmod_image(end_pc - 1) || !module_identity(tag, &mstart, &mend, &mid) ||
        end_pc > mend || mend - mstart > UINT_MAX) {
        s_skip[2]++;
        return NULL;
    }
    uint n = 0;
    instr_t *in;
    for (in = instrlist_first(ilist); in != NULL; in = instr_get_next(in))
        n++;
    if (n > MAX_INSTRS) {
        s_skip[3]++;
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
        s_skip[4]++;
        return NULL;
    }
    return s;
}

/* The block being recorded: its module, the addresses derived from its bytes, and the fixups
 * of its raw runs. Used under bb_building_lock.
 */
static struct {
    app_pc mstart, mend;
    app_pc derived[MAX_DERIVED];
    uint nderived;
    bool pin;
    int fail; /* F_* or -1 */
    struct fix {
        uint j, j_hi; /* snapshot instrs holding the value (j_hi: the high half, or -1) */
        uint pos, pos_hi;
        pc_val_t v;
    } fix[MAX_FIX];
    uint nfix;
    /* a 32-bit store that may be the low half of a 64-bit value */
    struct {
        bool valid;
        uint j, pos;
        uint lo;
        reg_id_t base, index, seg;
        int scale, disp;
    } prv;
} w;

static void
derive(app_pc a)
{
    if (w.nderived < MAX_DERIVED)
        w.derived[w.nderived++] = a;
    else
        w.fail = F_FIX;
}

/* Every address an instruction of the block names: its pc and end, rip-relative and branch
 * targets. Decoded from the recorded copy of the bytes.
 */
static void
derive_all(dcontext_t *dcontext, app_pc tag, const byte *src, uint len)
{
    instr_t tmp;
    w.nderived = 0;
    derive(tag);
    for (uint o = 0; o < len;) {
        instr_init(dcontext, &tmp);
        byte *next = decode_from_copy(dcontext, (byte *)src + o, tag + o, &tmp);
        app_pc t;
        if (next == NULL) {
            instr_free(dcontext, &tmp);
            w.fail = F_FIX;
            return;
        }
        o = (uint)(next - src);
        derive(tag + o);
        if (instr_get_rel_addr_target(&tmp, &t))
            derive(t);
        if (instr_is_cti(&tmp) && instr_num_srcs(&tmp) > 0 && opnd_is_pc(instr_get_target(&tmp)))
            derive(opnd_get_pc(instr_get_target(&tmp)));
        instr_free(dcontext, &tmp);
    }
}

static bool
is_derived(app_pc a)
{
    for (uint i = 0; i < w.nderived; i++) {
        if (w.derived[i] == a)
            return true;
    }
    return false;
}

static int
priv_index(app_pc a)
{
    for (uint i = 0; i < npriv; i++) {
        if (a >= priv[i].base && a < priv[i].base + priv[i].size)
            return (int)i;
    }
    return -1;
}

/* An immediate that is not a derived address: F_* if it may be an address, else -1. */
static int
mapped_class(app_pc a)
{
    if ((ptr_uint_t)a < 0x10000 || (ptr_uint_t)a >= 0x800000000000ULL)
        return -1; /* below mmap_min_addr, or not a user address */
    if (a >= get_dynamorio_dll_start() && a < get_dynamorio_dll_end())
        return F_IMM_DR;
    if (priv_index(a) >= 0)
        return F_IMM_PRIV;
    if (is_vmm_reserved_address(a, 1, NULL, NULL))
        return F_IMM_VMM;
    if (a >= w.mstart && a < w.mend)
        return F_IMM_MOD;
    os_get_module_info_lock();
    bool mod = module_pc_lookup(a) != NULL;
    os_get_module_info_unlock();
    if (mod)
        return F_IMM_MOD;
    dr_mem_info_t info;
    if (query_memory_ex(a, &info) && info.type != DR_MEMTYPE_FREE)
        return F_IMM_MAPPED;
    return -1;
}

/* A code address a pc-relative field or an exit names, as a value record. */
static bool
target_value(dcontext_t *dcontext, app_pc t, pc_val_t *v)
{
    ibl_type_t it;
    gencode_mode_t gm;
    int k;
    memset(v, 0, sizeof(*v));
    if (t >= w.mstart && t <= w.mend) {
        v->cls = V_MOD;
        v->off = t - w.mstart;
    } else if (t >= get_dynamorio_dll_start() && t < get_dynamorio_dll_end()) {
        v->cls = V_DR;
        v->off = t - get_dynamorio_dll_start();
    } else if ((k = priv_index(t)) >= 0) {
        v->cls = V_PRIV;
        v->arg = (ushort)k;
        v->off = t - priv[k].base;
    } else if (get_ibl_routine_type_ex(dcontext, t, &it, &gm)) {
        v->cls = V_IBL;
        v->arg = (ushort)(it.link_state | (it.source_fragment_type << 4) |
                          (it.branch_type << 8));
        v->off = gm;
        if (get_ibl_routine_ex(dcontext, it.link_state, it.source_fragment_type,
                               it.branch_type, gm) != t)
            return false;
    } else if (t == get_clean_call_save(dcontext, GENCODE_X64)) {
        v->cls = V_CCSAVE;
        v->off = GENCODE_X64;
    } else if (t == get_clean_call_restore(dcontext, GENCODE_X64)) {
        v->cls = V_CCRESTORE;
        v->off = GENCODE_X64;
    } else
        return false;
    v->pre = predicates(t);
    return true;
}

static void
add_fix(uint j, uint pos, uint j_hi, uint pos_hi, app_pc a)
{
    if (w.nfix == MAX_FIX) {
        w.fail = F_FIX;
        return;
    }
    struct fix *f = &w.fix[w.nfix++];
    f->j = j;
    f->pos = pos;
    f->j_hi = j_hi;
    f->pos_hi = pos_hi;
    memset(&f->v, 0, sizeof(f->v));
    f->v.cls = V_MOD;
    f->v.off = a - w.mstart;
    f->v.pre = predicates(a);
}

/* A 64-bit value at a known place: relocated if derived, else it must point nowhere. */
static void
value64(uint64 v, uint j, uint pos, uint j_hi, uint pos_hi)
{
    app_pc a = (app_pc)v;
    if (is_derived(a) && a >= w.mstart && a <= w.mend)
        add_fix(j, pos, j_hi, pos_hi, a);
    else if (is_derived(a))
        w.fail = F_IMM_MOD; /* an address the block names, outside its module */
    else {
        int c = mapped_class(a);
        if (c >= 0)
            w.fail = c;
    }
}

/* A value with no fixup: a derived one pins the entry to this base, a mapped one fails. */
static void
value_fixed(uint64 v)
{
    app_pc a = (app_pc)v;
    if (is_derived(a))
        w.pin = true;
    else {
        int c = mapped_class(a);
        if (c >= 0)
            w.fail = c;
    }
}

static void
flush_prv(void)
{
    if (w.prv.valid) {
        value_fixed((uint64)(int64)(int)w.prv.lo);
        value_fixed((uint64)w.prv.lo);
    }
    w.prv.valid = false;
}

/* The absolute values one emitted instruction holds. pos is where it starts within
 * snapshot instr j's bytes.
 */
static void
scan(dcontext_t *dcontext, instr_t *d, const byte *code, uint j, uint pos, uint ilen)
{
    int opc = instr_get_opcode(d);
    bool pair_lo = false;
    for (int i = 0; i < instr_num_srcs(d); i++) {
        opnd_t o = instr_get_src(d, i);
        if (!opnd_is_immed_int(o))
            continue;
        ptr_int_t v = opnd_get_immed_int(o);
        opnd_size_t sz = opnd_get_size(o);
        if (sz == OPSZ_8) {
            /* an immediate is the instruction's last field */
            if (ilen < 8 || rd64(code + ilen - 8) != (uint64)v) {
                value_fixed((uint64)v);
                continue;
            }
            value64((uint64)v, j, pos + ilen - 8, (uint)-1, 0xffff);
        } else if (sz == OPSZ_4) {
            bool at_end = ilen >= 4 && rd32(code + ilen - 4) == (uint)v;
            opnd_t m = instr_num_dsts(d) > 0 ? instr_get_dst(d, 0) : opnd_create_null();
            if (at_end && opc == OP_mov_st && opnd_is_base_disp(m) &&
                opnd_get_size(m) == OPSZ_4 && w.prv.valid &&
                w.prv.base == opnd_get_base(m) && w.prv.index == opnd_get_index(m) &&
                (w.prv.index == DR_REG_NULL || w.prv.scale == opnd_get_scale(m)) &&
                w.prv.seg == opnd_get_segment(m) &&
                w.prv.disp + 4 == opnd_get_disp(m)) {
                /* the high half of insert_mov_immed_arch/insert_push_immed_arch's pair */
                uint64 full = (uint64)w.prv.lo | ((uint64)(uint)v << 32);
                w.prv.valid = false;
                value64(full, w.prv.j, w.prv.pos, j, pos + ilen - 4);
                continue;
            }
            flush_prv();
            if (at_end && opc == OP_push_imm) {
                w.prv = (__typeof__(w.prv)) { true, j, pos + ilen - 4, (uint)v, DR_REG_XSP,
                                          DR_REG_NULL, DR_REG_NULL, 0, 0 };
                pair_lo = true;
            } else if (at_end && opc == OP_mov_st && opnd_is_base_disp(m) &&
                       opnd_get_size(m) == OPSZ_4) {
                w.prv = (__typeof__(w.prv)) { true, j, pos + ilen - 4, (uint)v,
                                          opnd_get_base(m), opnd_get_index(m),
                                          opnd_get_segment(m), opnd_get_scale(m),
                                          opnd_get_disp(m) };
                pair_lo = true;
            } else {
                value_fixed((uint64)(int64)(int)v);
                value_fixed((uint64)(uint)v);
            }
        }
    }
    if (!pair_lo)
        flush_prv(); /* a pending low half's next instruction was not its high half */
    /* absolute memory operands: no fixup, so they must point nowhere or be pinned */
    for (int k = 0; k < instr_num_srcs(d) + instr_num_dsts(d); k++) {
        opnd_t o = k < instr_num_srcs(d) ? instr_get_src(d, k)
                                         : instr_get_dst(d, k - instr_num_srcs(d));
        if (opnd_is_base_disp(o)) {
            reg_id_t seg = opnd_get_segment(o);
            if (opnd_get_base(o) == DR_REG_NULL && seg != DR_SEG_FS && seg != DR_SEG_GS)
                value_fixed((uint64)(int64)opnd_get_disp(o));
        } else if (opnd_is_abs_addr(o) && !opnd_is_rel_addr(o)) {
            reg_id_t seg = opnd_get_segment(o);
            if (seg != DR_SEG_FS && seg != DR_SEG_GS)
                value_fixed((uint64)opnd_get_addr(o));
        }
    }
}

/* A raw instr's emitted bytes: plain, with absolute values scanned, or one instruction with a
 * pc-relative field (APPREL or REL).
 */
static bool
classify(dcontext_t *dcontext, pcache_ref_t *s, uint j, byte *code, uint len)
{
    struct snap_ins *si = &s->ins[j];
    instr_t *in = si->in;
    instr_t tmp;
    bool ok = true;
    /* An app instruction emitted as its own bytes holds nothing position-dependent. */
    bool own = !instr_is_meta(in) && instr_raw_bits_valid(in) && in->length == len &&
        in->bytes >= s->tag && in->bytes + len <= s->end && memcmp(in->bytes, code, len) == 0;
    for (uint o = 0; o < len && ok;) {
        instr_init(dcontext, &tmp);
        byte *next = decode_from_copy(dcontext, code + o, code + o, &tmp);
        app_pc target = NULL;
        int pos = -1;
        uint nfix = w.nfix;
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
                si->kind = R_APPREL; /* replayed from the compared app bytes */
            else
                si->kind = R_REL;
        }
        if (next != NULL && ok) {
            if (own || si->kind == R_APPREL)
                flush_prv();
            else
                scan(dcontext, &tmp, code + o, j, o, (uint)(next - (code + o)));
            /* a REL record is re-encoded from operands: no place for a fixup */
            if (pos >= 0 && (w.nfix != nfix || w.prv.valid))
                ok = false;
        }
        instr_free(dcontext, &tmp);
        if (next != NULL)
            o = (uint)(next - code);
    }
    if (!ok)
        w.fail = F_FIX;
    return ok;
}

static byte wbuf[1 << 17];
static cache_pc wpcs[MAX_INSTRS];
static uint boff[MAX_INSTRS]; /* offset of a raw instr in its bundle */
static uint bnum[MAX_INSTRS]; /* its bundle */

static byte *
put_val(byte *p, const pc_val_t *v)
{
    memcpy(p, v, sizeof(*v));
    return p + sizeof(*v);
}

void
pcache_record(dcontext_t *dcontext, pcache_ref_t *s, instrlist_t *ilist, fragment_t *f)
{
    uint n = s->n, j = 0, nrec = 0;
    cache_pc pc = FCACHE_ENTRY_PC(f);
    instr_t *in;
    uint64 mid;
    bool ok = true;
    memset(&w.prv, 0, sizeof(w.prv));
    w.nfix = 0;
    w.pin = false;
    w.fail = -1;
    if (!module_identity(s->tag, &w.mstart, &w.mend, &mid))
        ok = false;
    /* Where each snapshot instr landed; pad nops emit inserted are not in the snapshot. */
    for (in = instrlist_first(ilist); in != NULL && ok; in = instr_get_next(in)) {
        if (j < n && in == s->ins[j].in)
            wpcs[j++] = pc;
        if (instr_ok_to_emit(in))
            pc += instr_length(dcontext, in);
    }
    if (j != n)
        ok = false;
    if (ok)
        derive_all(dcontext, s->tag, s->tag, (uint)(s->end - s->tag));
    for (j = 0; j < n && ok && w.fail < 0; j++) {
        if (s->ins[j].kind == R_RAW) {
            classify(dcontext, s, j, wpcs[j], instr_length(dcontext, s->ins[j].in));
        } else if (s->ins[j].kind != 0 || s->ins[j].is_target)
            flush_prv();
    }
    flush_prv();
    if (w.fail >= 0)
        ok = false;
    /* Record indices: raw instrs share a bundle unless one is a jump target. */
    bool open = false;
    uint off = 0;
    for (j = 0; j < n && ok; j++) {
        struct snap_ins *si = &s->ins[j];
        si->rec = nrec;
        if (si->kind == 0 && !si->is_target)
            continue;
        if (si->kind == R_RAW && open && !si->is_target) {
            boff[j] = off;
            bnum[j] = nrec - 1;
            off += instr_length(dcontext, si->in);
            continue;
        }
        open = si->kind == R_RAW;
        if (open) {
            boff[j] = 0;
            bnum[j] = nrec;
            off = instr_length(dcontext, si->in);
        }
        nrec++;
    }
    /* A value's two halves must share a bundle. */
    for (uint k = 0; k < w.nfix && ok; k++) {
        struct fix *x = &w.fix[k];
        if (s->ins[x->j].kind != R_RAW ||
            (x->j_hi != (uint)-1 &&
             (s->ins[x->j_hi].kind != R_RAW || bnum[x->j_hi] != bnum[x->j]))) {
            s_fail[F_FIX]++;
            ok = false;
        }
    }
    pc_entry_t *e = (pc_entry_t *)wbuf;
    byte *p = (byte *)(e + 1), *bundle = NULL, *rec0 = NULL;
    memset(e, 0, sizeof(*e));
    if (ok) {
        e->modid = mid;
        e->mod_off = (uint)(s->tag - w.mstart);
        e->mod_size = (uint)(w.mend - w.mstart);
        e->mod_start = (uint64)w.mstart;
        e->src_len = (ushort)(s->end - s->tag);
        e->initial_flags = s->initial_flags;
        e->flags = s->flags;
        e->avx512 = s->avx512 ? 3 : 0;
        e->eflags = w.pin ? E_PIN : 0;
        e->nrec = (ushort)nrec;
        memcpy(p, s->tag, e->src_len);
        p += e->src_len;
        rec0 = p;
    }
    uint bn = 0;
    for (j = 0; j <= n && ok; j++) {
        struct snap_ins *si = j < n ? &s->ins[j] : NULL;
        pc_val_t v;
        uint len;
        if (p + 64 + MAX_SRC + MAX_FIX * SZ_FIX > wbuf + sizeof(wbuf)) {
            ok = false;
            break;
        }
        /* A raw bundle ends before any other record and before a jump target: its fixups
         * follow its bytes.
         */
        if (bundle != NULL &&
            (si == NULL || si->is_target || (si->kind != R_RAW && si->kind != 0))) {
            ushort nf = 0;
            for (uint k = 0; k < w.nfix; k++) {
                struct fix *x = &w.fix[k];
                if (bnum[x->j] != bn)
                    continue;
                ushort pos = (ushort)(boff[x->j] + x->pos);
                ushort ph = x->j_hi == (uint)-1 ? 0xffff : (ushort)(boff[x->j_hi] + x->pos_hi);
                /* the writer's value is not stored: entries are the same in every layout */
                if (ph == 0xffff)
                    memset(bundle + SZ_RAW_HDR + pos, 0, 8);
                else {
                    memset(bundle + SZ_RAW_HDR + pos, 0, 4);
                    memset(bundle + SZ_RAW_HDR + ph, 0, 4);
                }
                memcpy(p, &pos, 2);
                memcpy(p + 2, &ph, 2);
                p = put_val(p + 4, &x->v);
                nf++;
            }
            memcpy(bundle + 2, &nf, 2);
            bundle = NULL;
        }
        if (si == NULL)
            break;
        switch (si->kind) {
        case 0:
            if (si->is_target)
                *p++ = R_LABEL;
            break;
        case R_EXIT:
            if (!target_value(dcontext, si->target, &v)) {
                s_fail[F_EXIT]++;
                ok = false;
                break;
            }
            *p++ = R_EXIT;
            memcpy(p, &si->opcode, 2);
            memcpy(p + 2, &si->prefixes, 4);
            memcpy(p + 6, &si->xflags, 4);
            p = put_val(p + 10, &v);
            break;
        case R_ICTI: {
            *p++ = R_ICTI;
            ushort t = (ushort)s->ins[si->tgt].rec;
            memcpy(p, &si->opcode, 2);
            memcpy(p + 2, &t, 2);
            p += SZ_ICTI;
            break;
        }
        case R_RAW: {
            len = instr_length(dcontext, si->in);
            if (bundle == NULL) {
                *p++ = R_RAW;
                bundle = p;
                bn = bnum[j];
                memset(p, 0, SZ_RAW_HDR);
                p += SZ_RAW_HDR;
            }
            memcpy(p, wpcs[j], len);
            p += len;
            ushort blen = rd16(bundle) + (ushort)len;
            memcpy(bundle, &blen, 2);
            break;
        }
        case R_APPREL: {
            *p++ = R_APPREL;
            ushort o = (ushort)(si->in->bytes - s->tag);
            memcpy(p, &o, 2);
            p[2] = (byte)si->in->length;
            p[3] = (byte)si->pos;
            p += SZ_APPREL;
            break;
        }
        case R_REL:
            len = instr_length(dcontext, si->in);
            if (!target_value(dcontext, si->target, &v) || len > 17) {
                s_fail[F_REL]++;
                ok = false;
                break;
            }
            *p++ = R_REL;
            p[0] = (byte)len;
            p[1] = (byte)si->pos;
            put_val(p + 2, &v);
            memcpy(p + SZ_REL_HDR, wpcs[j], len);
            memset(p + SZ_REL_HDR + si->pos, 0, 4);
            p += SZ_REL_HDR + len;
            break;
        }
    }
    if (ok && w.fail < 0) {
        e->rec_len = (uint)(p - rec0);
        size_t len = p - wbuf;
        e->hash = hash64(wbuf + sizeof(e->hash), len - sizeof(e->hash));
        size_t sz = ALIGN_FORWARD(len, 8);
        spin_lock();
        if (wbytes + sz + sizeof(pc_index_t) > DYNAMO_OPTION(pcache_max_size)) {
            s_cap++;
        } else {
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
            /* [size_t alloc size][size_t entry length][entry, padded] */
            byte *copy = (byte *)global_heap_alloc(sz + 2 * sizeof(size_t) HEAPACCT(ACCT_OTHER));
            ((size_t *)copy)[0] = sz;
            ((size_t *)copy)[1] = len;
            memset(copy + 2 * sizeof(size_t) + len, 0, sz - len);
            memcpy(copy + 2 * sizeof(size_t), wbuf, len);
            wents[wnum++] = copy;
            wbytes += sz + sizeof(pc_index_t);
            s_rec++;
        }
        spin_unlock();
    } else {
        if (w.fail >= 0)
            s_fail[w.fail]++;
        s_skip[5]++;
    }
    global_heap_free(s->ins, n * sizeof(*s->ins) HEAPACCT(ACCT_OTHER));
    global_heap_free(s, sizeof(*s) HEAPACCT(ACCT_OTHER));
}

static const pc_entry_t *
ent(byte *b)
{
    return (const pc_entry_t *)(b + 2 * sizeof(size_t));
}

static bool
ent_less(byte *a, byte *b)
{
    const pc_entry_t *x = ent(a), *y = ent(b);
    return x->modid < y->modid || (x->modid == y->modid && x->mod_off < y->mod_off);
}

/* Heap sort of the entries by (module, offset). */
static void
sift(byte **a, uint i, uint n)
{
    for (;;) {
        uint c = 2 * i + 1;
        if (c >= n)
            return;
        if (c + 1 < n && ent_less(a[c], a[c + 1]))
            c++;
        if (!ent_less(a[i], a[c]))
            return;
        byte *t = a[i];
        a[i] = a[c];
        a[c] = t;
        i = c;
    }
}

/* XXH64 over a stream, for the body written in pieces. */
typedef struct {
    uint64 v[4], total;
    byte mem[32];
    uint nmem;
} xstate_t;

static void
xinit(xstate_t *x)
{
    memset(x, 0, sizeof(*x));
    x->v[0] = XP1 + XP2;
    x->v[1] = XP2;
    x->v[3] = 0 - XP1;
}

static void
xupdate(xstate_t *x, const byte *p, size_t n)
{
    x->total += n;
    if (x->nmem + n < 32) {
        memcpy(x->mem + x->nmem, p, n);
        x->nmem += (uint)n;
        return;
    }
    if (x->nmem > 0) {
        uint k = 32 - x->nmem;
        memcpy(x->mem + x->nmem, p, k);
        for (int i = 0; i < 4; i++)
            x->v[i] = xround(x->v[i], rd64(x->mem + 8 * i));
        p += k;
        n -= k;
        x->nmem = 0;
    }
    for (; n >= 32; p += 32, n -= 32) {
        for (int i = 0; i < 4; i++)
            x->v[i] = xround(x->v[i], rd64(p + 8 * i));
    }
    memcpy(x->mem, p, n);
    x->nmem = (uint)n;
}

static uint64
xdigest(xstate_t *x)
{
    uint64 h;
    const byte *p = x->mem, *end = p + x->nmem;
    if (x->total >= 32) {
        h = rotl64(x->v[0], 1) + rotl64(x->v[1], 7) + rotl64(x->v[2], 12) +
            rotl64(x->v[3], 18);
        for (int i = 0; i < 4; i++)
            h = xmerge(h, x->v[i]);
    } else
        h = XP5;
    h += x->total;
    while (p + 8 <= end) {
        h ^= xround(0, rd64(p));
        h = rotl64(h, 27) * XP1 + XP4;
        p += 8;
    }
    if (p + 4 <= end) {
        h ^= (uint64)rd32(p) * XP1;
        h = rotl64(h, 23) * XP2 + XP3;
        p += 4;
    }
    while (p < end) {
        h ^= (*p) * XP5;
        h = rotl64(h, 11) * XP1;
        p++;
    }
    h ^= h >> 33;
    h *= XP2;
    h ^= h >> 29;
    h *= XP3;
    h ^= h >> 32;
    return h;
}

/* The body goes out twice: first only hashed, then written after the header. */
typedef struct {
    file_t f; /* INVALID_FILE: hash only */
    bool ok;
    size_t fill;
    xstate_t x;
} out_t;

static void
out_flush(out_t *o)
{
    if (o->fill > 0 && o->f != INVALID_FILE && os_write(o->f, wbuf, o->fill) != (ssize_t)o->fill)
        o->ok = false;
    o->fill = 0;
}

static void
out_put(out_t *o, const void *b, size_t n)
{
    const byte *q = (const byte *)b;
    if (o->f == INVALID_FILE) {
        xupdate(&o->x, q, n);
        return;
    }
    while (n > 0) {
        if (o->fill == sizeof(wbuf))
            out_flush(o);
        size_t k = MIN(n, sizeof(wbuf) - o->fill);
        memcpy(wbuf + o->fill, q, k);
        o->fill += k;
        q += k;
        n -= k;
    }
}

static void
out_body(out_t *o, const pc_header_t *h, uint num)
{
    static const byte zero[8];
    out_put(o, key, key_len);
    out_put(o, zero, h->index_off - h->key_off - key_len);
    uint64 off = 0;
    for (uint i = 0; i < num; i++) {
        const pc_entry_t *e = ent(wents[i]);
        pc_index_t x = { e->modid, e->mod_off, (uint)((size_t *)wents[i])[1], off };
        out_put(o, &x, sizeof(x));
        off += ((size_t *)wents[i])[0];
    }
    for (uint i = 0; i < num; i++)
        out_put(o, ent(wents[i]), ((size_t *)wents[i])[0]);
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
    /* drop later builds of a block (rebuilt after a flush) */
    for (i = 0; i < wnum; i++) {
        if (num == 0 || ent_less(wents[num - 1], wents[i]))
            wents[num++] = wents[i];
    }
    pc_header_t h;
    memset(&h, 0, sizeof(h));
    h.magic = PC_MAGIC;
    h.version = PC_VERSION;
    h.header_size = sizeof(h);
    h.num = num;
    h.key_off = sizeof(h);
    h.key_len = key_len;
    h.index_off = ALIGN_FORWARD(h.key_off + key_len, 8);
    h.data_off = h.index_off + (uint64)num * sizeof(pc_index_t);
    for (i = 0; i < num; i++)
        h.data_len += ((size_t *)wents[i])[0];
    h.file_size = h.data_off + h.data_len;
    out_t o;
    memset(&o, 0, sizeof(o));
    o.f = INVALID_FILE;
    xinit(&o.x);
    out_body(&o, &h, num);
    h.body_hash = xdigest(&o.x);
    h.header_hash = hash64(&h, sizeof(h));
    cache_path(path, sizeof(path));
    snprintf(tmp, sizeof(tmp), "%s.%d.tmp", path, get_process_id());
    tmp[sizeof(tmp) - 1] = '\0';
    o.ok = true;
    o.fill = 0;
    o.f = os_open(tmp, OS_OPEN_WRITE | OS_OPEN_REQUIRE_NEW);
    if (o.f == INVALID_FILE) {
        print_file(STDERR, "pcache: cannot write %s\n", tmp);
        return;
    }
    out_put(&o, &h, sizeof(h));
    out_body(&o, &h, num);
    out_flush(&o);
    os_close(o.f);
    if (o.ok && os_rename_file(tmp, path, true)) {
        if (DYNAMO_OPTION(pcache_stats))
            print_file(STDERR, "pcache: wrote %u entries, %llu bytes, to %s\n", num,
                       (unsigned long long)h.file_size, path);
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
        write_file();
        spin_unlock();
    }
    if (DYNAMO_OPTION(pcache_stats)) {
        print_file(STDERR,
                   "pcache: pid %d mode %d lookups %u hits %u noentry %u corrupt %u "
                   "iflags %u avx %u module %u pin %u area %u bytes %u build %u pre %u "
                   "verify_ok %u verify_bad %u recorded %u cap %u "
                   "skip %u/%u/%u/%u/%u/%u fail dr %u priv %u vmm %u mod %u mapped %u "
                   "rel %u exit %u fix %u build_by_kind %u/%u/%u/%u/%u/%u/%u\n",
                   get_process_id(), mode, s_lookup, s_hit, s_noentry, s_corrupt, s_iflags,
                   s_avx, s_module, s_pin, s_area, s_bytes, s_build, s_pre, s_vok, s_vbad,
                   s_rec, s_cap, s_skip[0], s_skip[1], s_skip[2], s_skip[3], s_skip[4],
                   s_skip[5], s_fail[F_IMM_DR], s_fail[F_IMM_PRIV], s_fail[F_IMM_VMM],
                   s_fail[F_IMM_MOD], s_fail[F_IMM_MAPPED], s_fail[F_REL], s_fail[F_EXIT],
                   s_fail[F_FIX], s_bad_kind[0], s_bad_kind[1], s_bad_kind[2], s_bad_kind[3],
                   s_bad_kind[4], s_bad_kind[5], s_bad_kind[6]);
    }
}
