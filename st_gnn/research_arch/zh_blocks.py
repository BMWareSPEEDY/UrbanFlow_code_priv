import struct

data = open('swmm_b2_ahmedabad_scen0.out', 'rb').read()
n = len(data)

def find_u32(val, start=0):
    pat = struct.pack('<I', val)
    idx = data.find(pat, start)
    return idx

# find positions of u32==60 (step time?) after the ID region (~offset 100000+)
first60 = find_u32(60, 100000)
print('first u32=60 at', first60)
positions = []
p = first60
while p != -1 and len(positions) < 12:
    positions.append(p)
    p = find_u32(60, p + 1)
print('positions of 60:', positions)
if len(positions) > 1:
    gaps = [positions[i+1] - positions[i] for i in range(len(positions)-1)]
    print('gaps:', gaps, '-> block size', gaps[0] if gaps else None)

# also check what precedes a 60 block
off = positions[0]
print('before first 60:', data[off-32:off].hex())
print('after  first 60:', data[off:off+64].hex())
print('u32 at off-8, off-4:', struct.unpack_from('<I', data, off-8)[0], struct.unpack_from('<I', data, off-4)[0])

# check total size of the payload after IDs: where do float-ish zeros begin?
# count u32 60 overall
cnt60 = 0
p = 0
while True:
    p = data.find(struct.pack('<I', 60), p + 4 if p else 0)
    if p == -1:
        break
    cnt60 += 1
    p += 4
print('total u32==60 count:', cnt60)