import struct
import sys

def probe(path):
    with open(path, 'rb') as f:
        magic, = struct.unpack('<H', f.read(2))
        if magic != 0xDE0F:
            print(f"{path}: bad magic {magic:#x}"); return
        f.read(8)  # "SWMM5" + nulls
        ver, = struct.unpack('<i', f.read(4))
        units, = struct.unpack('<i', f.read(4))
        npoll, = struct.unpack('<i', f.read(4))
        nsub, = struct.unpack('<i', f.read(4))
        nnode, = struct.unpack('<i', f.read(4))
        nlink, = struct.unpack('<i', f.read(4))
        nsteps, = struct.unpack('<i', f.read(4))
        rstep, = struct.unpack('<i', f.read(4))
        f.read(8)  # start date/time (2 ints)
        dur, = struct.unpack('<i', f.read(4))
        print(f"{path}: ver {ver} units {units} npoll {npoll} nsub {nsub} nnode {nnode} nlink {nlink} "
              f"nsteps {nsteps} report_step {rstep}s sim_dur {dur}s")

if __name__ == '__main__':
    for p in sys.argv[1:]:
        probe(p)