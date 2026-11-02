"""Download the first N bytes of a remote zip and extract every complete entry in them.

The Monash labelled test frames sit at the start of a 9.8 GB archive, so the first ~330 MB
holds about 100 labelled frames without fetching the rest.
    python tools/fetch_zip_prefix.py <url> <bytes> <outdir>
"""
import os
import struct
import sys
import urllib.request
import zlib

url, nbytes, outdir = sys.argv[1], int(sys.argv[2]), sys.argv[3]
buf, got = bytearray(), 0
while got < nbytes:  # the server drops long connections now and then; resume where it stopped
    req = urllib.request.Request(url, headers={"Range": f"bytes={got}-{nbytes - 1}"})
    with urllib.request.urlopen(req, timeout=60) as r:
        while chunk := r.read(1 << 22):
            buf += chunk
            got += len(chunk)
p = n = 0
while p + 30 <= len(buf) and buf[p:p + 4] == b"PK\x03\x04":
    flag, comp = struct.unpack("<HH", buf[p + 6:p + 10])
    csize = struct.unpack("<I", buf[p + 18:p + 22])[0]
    nlen, elen = struct.unpack("<HH", buf[p + 26:p + 30])
    name = buf[p + 30:p + 30 + nlen].decode()
    s = p + 30 + nlen + elen
    if s + csize > len(buf):
        break
    if not name.endswith("/"):
        out = os.path.join(outdir, name)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        raw = bytes(buf[s:s + csize])
        open(out, "wb").write(zlib.decompress(raw, -15) if comp == 8 else raw)
        n += 1
    p = s + csize + (16 if flag & 8 else 0)
print("extracted", n, "files")
