import struct
import sys

def parse(path):
    data = open(path, 'rb').read()
    n = len(data)
    print(f"{path}: {n} bytes")
    print("--- header u32s ---")
    for i in range(0, 64, 4):
        v = struct.unpack_from('<I', data, i)[0]
        print(f"  off {i:4d}: {v} ({v:#x})")
    print("--- strings region ---")
    pos = 32
    nnode = None
    nlink = None
    nsteps = None
    ids = []
    for k in range(6):
        ln = struct.unpack_from('<I', data, pos)[0]
        if ln == 0 or ln > 100:
            print(f"  off {pos}: len {ln} -- end of strings?")
            break
        s = data[pos+4:pos+4+ln]
        ids.append((pos, ln, s))
        print(f"  off {pos}: len {ln} str {s!r}")
        pos += 4 + ln
        if k == 0:
            nnode = ln  # first string is likely node 0

    # try find a repeated structure: scan for plausibly increasing u32 timestamps
    print("--- searching for step timestamps (60,120,...) ---")
    import re
    cands = []
    for base in [60, 30, 15]:
        pat = struct.pack('<I', base)
        idx = data.find(pat, pos)
        if idx > 0:
            cands.append((base, idx))
    print("  candidates:", cands)

    # heuristic: count occurrences of each u32 in tail half, look for the most common
    from collections import Counter
    tail = data[n//2:]
    cnt = Counter()
    for i in range(0, len(tail) - 4, 4):
        cnt[struct.unpack_from('<I', tail, i)[0]] += 1
    common = cnt.most_common(5)
    print("  most common u32 in tail:", [(v, c) for v, c in common])

if __name__ == '__main__':
    for p in sys.argv[1:]:
        parse(p)
        print()