"""Download the paper's data records from Zenodo, check them and unpack them under one data root.

    python -m reproduce.fetch --root DATA [--records NAME ...] [--archives NAME ...] [--keep-archives] [--draft]

Each record holds archives (``*.tar.zst``) of files under the data root. The tables record holds the tables by licence
(``tables-cc-by-4.0``, ``tables-cc-by-nc-4.0``). The inputs record holds the inputs by licence (``inputs-cc-by-4.0``,
``inputs-2-cc-by-nc-4.0``, ...), one archive per synthetic-answer instance (``assay-instance-tahoe-100m-F1``, ...), the
information ladder's saved scores (``tahoe-100m-ladder-01``, ``parse-pbmc-ladder-01``, ...), the run records of
information localization (``information-localization-run-records``) and the one case of origin resolution
(``replogle-nadig-jiang-one-case``). The saved predictions are in three records, in archives named after the screen
they hold (``gse306429-01``, ``genetic-screens-predictions-01``, ``tahoe-100m-probes-01``, ``tahoe-100m-repairs-01``,
...). An archive is checked against the MD5 that Zenodo publishes for it, unpacked with ``zstd -d --long=31`` into
``tar``, and deleted unless ``--keep-archives`` is given. Then the files the manifest lists for the record are checked for their size and
SHA-256: every one of them when the whole record was fetched; with ``--archives``, which fetches only the archives
whose names start with one of the given names (``replogle-nadig-jiang-one-case``, ``gse306429``), those that are under
the data root. ``zstd`` (1.4 or later) and ``tar`` must be on the path; unpacking takes up to 2 GB of memory.

``--draft`` reads the records before they are published: their drafts, shared with the reader's Zenodo account, through
a personal access token of that account in the environment variable ``ZENODO_TOKEN``.
"""
import argparse
import csv
import hashlib
import json
import os
import subprocess
import urllib.request
from pathlib import Path

__all__ = ["RECORDS", "MANIFEST", "record_files", "download", "unpack", "check", "main"]

# record name -> the Zenodo id of the version the paper describes
RECORDS = {"tables": 23997774, "inputs": 24210211, "other-screens": 24212188,
           "tahoe-100m-control-use-and-probes": 24212190, "tahoe-100m-repairs": 24212192}
MANIFEST = Path(__file__).with_name("data") / "MANIFEST.tsv"
API = "https://zenodo.org/api/records/{}"
CHUNK = 1 << 24


def request(url, token=None):
    """A GET request, with the access token when one is given."""
    return urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"} if token else {})


def record_files(record_id, token=None):
    """(name, size, md5, url) of every file of a Zenodo record, or of its draft when a ``token`` is given."""
    if token is None:
        with urllib.request.urlopen(request(API.format(record_id))) as response:
            files = json.load(response)["files"]
        return [(f["key"], f["size"], f["checksum"].removeprefix("md5:"), f["links"]["self"]) for f in files]
    with urllib.request.urlopen(request(API.format(record_id) + "/draft/files", token)) as response:
        files = json.load(response)["entries"]
    return [(f["key"], f["size"], f["checksum"].removeprefix("md5:"), f["links"]["content"]) for f in files]


def digest(path, algorithm):
    h = hashlib.new(algorithm)
    with open(path, "rb") as f:
        while block := f.read(CHUNK):
            h.update(block)
    return h.hexdigest()


def download(url, target, md5, token=None):
    """Download ``url`` to ``target`` and check its MD5; the file appears under its name only when it matches."""
    part = target.with_name(target.name + ".part")
    with urllib.request.urlopen(request(url, token)) as response, open(part, "wb") as f:
        while block := response.read(CHUNK):
            f.write(block)
    if digest(part, "md5") != md5:
        raise ValueError(f"{target.name}: MD5 differs from the record's")
    part.rename(target)


def unpack(archive, root):
    """Unpack a ``.tar.zst`` archive under ``root``."""
    zstd = subprocess.Popen(["zstd", "-d", "--long=31", "-c", str(archive)], stdout=subprocess.PIPE)
    tar = subprocess.run(["tar", "-x", "-C", str(root)], stdin=zstd.stdout)
    zstd.stdout.close()
    if zstd.wait() or tar.returncode:
        raise RuntimeError(f"{archive.name}: unpacking failed")


def check(root, record, whole=True):
    """The files the manifest lists for ``record``, by size and SHA-256: all of them, or with ``whole=False`` those under
    ``root``; returns the number checked."""
    with open(MANIFEST, newline="") as f:
        rows = [r for r in csv.DictReader(f, delimiter="\t") if r["record"] == record and (whole or (root / r["path"]).exists())]
    bad = [r["path"] for r in rows if not (root / r["path"]).is_file() or (root / r["path"]).stat().st_size != int(r["bytes"])
           or digest(root / r["path"], "sha256") != r["sha256"]]
    if bad:
        raise ValueError(f"{record}: {len(bad)} of {len(rows)} files missing or different, first {bad[0]}")
    return len(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Download, check and unpack the paper's data records from Zenodo.")
    parser.add_argument("--root", type=Path, required=True, help="the data root the scripts read")
    parser.add_argument("--records", nargs="+", choices=list(RECORDS), default=list(RECORDS))
    parser.add_argument("--archives", nargs="+", help="only the archives whose names start with one of these")
    parser.add_argument("--keep-archives", action="store_true")
    parser.add_argument("--draft", action="store_true", help="read the unpublished drafts (token in ZENODO_TOKEN)")
    args = parser.parse_args(argv)
    token = os.environ["ZENODO_TOKEN"] if args.draft else None
    archives = args.root / "archives"
    archives.mkdir(parents=True, exist_ok=True)
    fetched = 0
    for record in args.records:
        files = [f for f in record_files(RECORDS[record], token)
                 if not args.archives or f[0].startswith(tuple(args.archives))]
        if not files:
            continue
        fetched += len(files)
        for name, size, md5, url in files:
            target = archives / name
            if not target.exists():
                print(f"{record}: downloading {name} ({size / 1e9:.1f} GB)", flush=True)
                download(url, target, md5, token)
            if name.endswith(".tar.zst"):
                unpack(target, args.root)
                if not args.keep_archives:
                    target.unlink()
        print(f"{record}: {check(args.root, record, whole=not args.archives)} files checked", flush=True)
    if not fetched:
        parser.error(f"no archive of {', '.join(args.records)} starts with {', '.join(args.archives)}")


if __name__ == "__main__":
    main()
