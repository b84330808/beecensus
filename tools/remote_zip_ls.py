# List a remote zip's entries via HTTP Range (no full download).
import sys, io, struct, zipfile, urllib.request
url = sys.argv[1]
def get(rng):
    req = urllib.request.Request(url, headers={"Range": f"bytes={rng}"})
    with urllib.request.urlopen(req) as r:
        return r.read(), r.headers.get("Content-Range")
tail, cr = get("-200000")
total = int(cr.split("/")[1])
i = tail.rfind(b"PK\x05\x06")
cd_size, cd_off = struct.unpack("<II", tail[i+12:i+20])
if cd_off == 0xFFFFFFFF:  # zip64
    j = tail.rfind(b"PK\x06\x06")
    cd_size, cd_off = struct.unpack("<QQ", tail[j+40:j+56])
cd, _ = get(f"{cd_off}-{cd_off+cd_size-1}")
p = 0
sys.stdout.reconfigure(encoding="utf-8")
print("total", total)
while p < len(cd) and cd[p:p+4] == b"PK\x01\x02":
    (comp, csize, usize, nlen, elen, clen) = struct.unpack("<10xH8xIIHHH", cd[p:p+34])
    loff = struct.unpack("<I", cd[p+42:p+46])[0]
    name = cd[p+46:p+46+nlen].decode("utf-8", "replace")
    extra = cd[p+46+nlen:p+46+nlen+elen]
    # zip64 extra
    q = 0
    while q + 4 <= len(extra):
        hid, hlen = struct.unpack("<HH", extra[q:q+4])
        if hid == 1:
            vals = list(struct.unpack("<" + "Q"*(hlen//8), extra[q+4:q+4+hlen]))
            if usize == 0xFFFFFFFF: usize = vals.pop(0)
            if csize == 0xFFFFFFFF: csize = vals.pop(0)
            if loff == 0xFFFFFFFF: loff = vals.pop(0)
        q += 4 + hlen
    print(f"{name}\t{usize/1e6:.1f}MB\tcomp={comp}\tcsize={csize}\toff={loff}")
    p += 46 + nlen + elen + clen
