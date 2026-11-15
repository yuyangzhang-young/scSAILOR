"""Download the paper's data records from Zenodo, check them and unpack them under one data root.

    python -m reproduce.fetch --root DATA [--records NAME ...] [--archives NAME ...] [--keep-archives] [--local DIR]

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

``--local DIR`` downloads nothing: it unpacks and checks the archives already saved in ``DIR``, for example those
downloaded by hand from a record's page or from the share link of a record not yet published. Which archives belong to
a record is read from the manifest; an archive missing from ``DIR`` stops the command, and the archives are kept.
"""
import argparse
import csv
import hashlib
import json
import subprocess
import urllib.request
from pathlib import Path

__all__ = ["RECORDS", "MANIFEST", "record_files", "manifest_archives", "download", "unpack", "check", "main"]

# record name -> the Zenodo id of the version the paper describes
RECORDS = {"tables": 23997774, "inputs": 24210211, "other-screens": 24212188,
           "tahoe-100m-control-use-and-probes": 24212190, "tahoe-100m-repairs": 24212192}
MANIFEST = Path(__file__).with_name("data") / "MANIFEST.tsv"
API = "https://zenodo.org/api/records/{}"
CHUNK = 1 << 24


def manifest_rows():
    with open(MANIFEST, newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def record_files(record_id):
    """(name, size, md5, url) of every file of a published Zenodo record."""
    with urllib.request.urlopen(API.format(record_id)) as response:
        files = json.load(response)["files"]
    return [(f["key"], f["size"], f["checksum"].removeprefix("md5:"), f["links"]["self"]) for f in files]


def manifest_archives(record):
    """The names of the archives the manifest lists for ``record``."""
    return sorted({r["archive"] for r in manifest_rows() if r["record"] == record})


def digest(path, algorithm):
    h = hashlib.new(algorithm)
    with open(path, "rb") as f:
        while block := f.read(CHUNK):
            h.update(block)
    return h.hexdigest()


def download(url, target, md5):
    """Download ``url`` to ``target`` and check its MD5; the file appears under its name only when it matches."""
    part = target.with_name(target.name + ".part")
    with urllib.request.urlopen(url) as response, open(part, "wb") as f:
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
    rows = [r for r in manifest_rows() if r["record"] == record and (whole or (root / r["path"]).exists())]
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
    parser.add_argument("--local", type=Path, metavar="DIR",
                        help="download nothing; unpack and check the archives already saved in DIR")
    args = parser.parse_args(argv)
    args.root.mkdir(parents=True, exist_ok=True)
    archives = args.local or args.root / "archives"
    archives.mkdir(exist_ok=True)
    fetched = 0
    for record in args.records:
        if args.local:
            files = [(name, None, None, None) for name in manifest_archives(record)]
        else:
            files = record_files(RECORDS[record])
        files = [f for f in files if not args.archives or f[0].startswith(tuple(args.archives))]
        if not files:
            continue
        fetched += len(files)
        for name, size, md5, url in files:
            target = archives / name
            if not target.exists():
                if args.local:
                    parser.error(f"{record}: {name} is not in {archives}")
                print(f"{record}: downloading {name} ({size / 1e9:.1f} GB)", flush=True)
                download(url, target, md5)
            if name.endswith(".tar.zst"):
                unpack(target, args.root)
                if not (args.keep_archives or args.local):
                    target.unlink()
        print(f"{record}: {check(args.root, record, whole=not args.archives)} files checked", flush=True)
    if not fetched:
        parser.error(f"no archive of {', '.join(args.records)} starts with {', '.join(args.archives)}")


if __name__ == "__main__":
    main()
