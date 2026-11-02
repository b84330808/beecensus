# Inflate one zip entry over HTTP Range, resuming when the server drops the connection.
import sys, zlib, struct, urllib.request, time
url, off, csize, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
hdr = urllib.request.urlopen(urllib.request.Request(url, headers={"Range": f"bytes={off}-{off+29}"})).read()
nlen, elen = struct.unpack("<HH", hdr[26:30])
start = off + 30 + nlen + elen
d = zlib.decompressobj(-15); got = 0; tries = 0
with open(out, "wb") as f:
    while got < csize and not d.eof:
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={start+got}-{start+csize-1}"})
            with urllib.request.urlopen(req, timeout=60) as r:
                while True:
                    c = r.read(1 << 20)
                    if not c: break
                    got += len(c); f.write(d.decompress(c))
        except Exception as ex:
            tries += 1; print("resume after", got, ex, flush=True); time.sleep(3)
            if tries > 30: raise
        if got < csize and not d.eof: print("short read at", got, "of", csize, flush=True)
print("ok", out, got, csize, d.eof)
