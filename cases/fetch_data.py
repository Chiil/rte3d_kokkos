"""Fetch the case files that are too large to keep in git.

    python cases/fetch_data.py les_cloudfield              # the input, 26 MB
    python cases/fetch_data.py les_cloudfield --reference  # and the reference, 149 MB

Only the cases listed in DATASETS below need this; the rest either carry their input
in extern/rrtmgp-data or generate it. The download is a deliberate step rather than
something a run script does behind your back, since these files are tens of megabytes.

Nothing here is a dependency of rte3d: it is the standard library, and the files it
writes are ordinary NetCDF that cases/run_case.py and xarray read.
"""
import argparse
import hashlib
import os
import sys
import urllib.request

CASES = os.path.dirname(os.path.abspath(__file__))

# Zenodo serves a file's bytes from the record id and the file name, which is what the
# record's own API reports as that file's link.
ZENODO_FILE = 'https://zenodo.org/api/records/{record}/files/{name}/content'


class File:
    """One file to fetch, and where it belongs once it is here.

    record is the version's Zenodo id rather than the concept DOI's, so that a new
    version published upstream does not silently change what a case solves. md5 is the
    checksum that record publishes, and is what says the file arrived intact.

    reference marks a file a case does not need in order to run -- something to compare
    against afterwards -- so that fetching the case does not pull it by default.
    """

    def __init__(self, record, name, md5, size, dest, what, reference=False):
        self.record = record
        self.name = name
        self.md5 = md5
        self.size = size
        self.dest = os.path.join(CASES, *dest)
        self.what = what
        self.reference = reference

    @property
    def url(self):
        return ZENODO_FILE.format(record=self.record, name=self.name)


class Dataset:
    """Everything one case fetches, and the archive it comes from."""

    def __init__(self, doi, license, files):
        self.doi = doi
        self.license = license
        self.files = files

    def wanted(self, reference):
        return [f for f in self.files if reference or not f.reference]

    @property
    def size(self):
        return sum(f.size for f in self.files if not f.reference)


LES_CLOUDFIELD = 18757089

DATASETS = {
    'les_cloudfield': Dataset(
        doi='10.5281/zenodo.18757088',
        license='CC BY 4.0',
        files=[
            File(record=LES_CLOUDFIELD,
                 name='test_input.nc',
                 md5='b8e7b24c786175bc9addcd824cadfbd6',
                 size=25956852,
                 dest=('les_cloudfield', 'les_cloudfield_input.nc'),
                 what='the RICO cumulus field, 128 x 128 x 200 cells'),
            File(record=LES_CLOUDFIELD,
                 name='test_output_forward.nc',
                 md5='ea0484506d6df4f5557d6621f2ab98f6',
                 size=148924103,
                 dest=('les_cloudfield', 'les_cloudfield_reference.nc'),
                 what="rte-rrtmgp-cpp's own fluxes for this field",
                 reference=True),
        ],
    ),
}


def md5sum(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(chunk), b''):
            h.update(block)

    return h.hexdigest()


def fetch_file(dataset, f, force=False):
    """Download one file unless it is already here and intact. Returns its path."""
    if os.path.exists(f.dest) and not force:
        if md5sum(f.dest) == f.md5:
            print(f'{f.dest} is already here')
            return f.dest
        print(f'{f.dest} does not match its checksum; fetching it again')

    os.makedirs(os.path.dirname(f.dest), exist_ok=True)

    print(f'{dataset.doi}\n  {f.url}\n  -> {f.dest} ({f.size/1e6:.0f} MB)')

    # Written beside the destination and moved into place at the end, so that an
    # interrupted download never leaves something a case would go on to read.
    partial = f.dest + '.part'
    try:
        with urllib.request.urlopen(f.url) as response, open(partial, 'wb') as out:
            done = 0
            while block := response.read(1 << 20):
                out.write(block)
                done += len(block)
                if sys.stderr.isatty():
                    print(f'\r  {done/1e6:6.1f} / {f.size/1e6:.0f} MB',
                          end='', file=sys.stderr)
        if sys.stderr.isatty():
            print(file=sys.stderr)

        got = md5sum(partial)
        if got != f.md5:
            raise SystemExit(f'checksum mismatch: expected {f.md5}, got {got}')

        os.replace(partial, f.dest)
    finally:
        if os.path.exists(partial):
            os.remove(partial)

    print(f'wrote {f.dest}')

    return f.dest


def fetch(dataset, reference=False, force=False):
    return [fetch_file(dataset, f, force) for f in dataset.wanted(reference)]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('case', nargs='?', choices=sorted(DATASETS), help='which case')
    p.add_argument('--reference', action='store_true',
                   help='also fetch the fluxes to compare against, where there are any')
    p.add_argument('--force', action='store_true',
                   help='download again even if the file is already here')
    args = p.parse_args()

    if args.case is None:
        print('Cases whose data is fetched rather than stored:\n')
        for name, d in sorted(DATASETS.items()):
            print(f'  {name} -- {d.doi}, {d.license}')
            for f in d.files:
                tag = '  (--reference)' if f.reference else ''
                print(f'    {f.size/1e6:5.0f} MB  {f.what}{tag}')
        return

    fetch(DATASETS[args.case], reference=args.reference, force=args.force)


if __name__ == '__main__':
    main()
