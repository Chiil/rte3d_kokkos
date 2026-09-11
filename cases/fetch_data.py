"""Fetch the case input files that are too large to keep in git.

    python cases/fetch_data.py les_cloudfield

Only the cases listed in DATASETS below need this; the rest either carry their input
in extern/rrtmgp-data or generate it. The download is a deliberate step rather than
something a run script does behind your back, since these files are tens of megabytes.

Nothing here is a dependency of rte3d: it is the standard library, and the file it
writes is an ordinary NetCDF input that cases/user/run_case.py reads.
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


class Dataset:
    """One file to fetch, and where it belongs once it is here.

    record is the version's Zenodo id rather than the concept DOI's, so that a new
    version published upstream does not silently change what a case solves. md5 is the
    checksum that record publishes, and is what says the file arrived intact.
    """

    def __init__(self, record, name, md5, size, dest, doi, what):
        self.record = record
        self.name = name
        self.md5 = md5
        self.size = size
        self.dest = os.path.join(CASES, *dest)
        self.doi = doi
        self.what = what

    @property
    def url(self):
        return ZENODO_FILE.format(record=self.record, name=self.name)


DATASETS = {
    'les_cloudfield': Dataset(
        record=18757089,
        name='test_input.nc',
        md5='b8e7b24c786175bc9addcd824cadfbd6',
        size=25956852,
        dest=('les_cloudfield', 'les_cloudfield_input.nc'),
        doi='10.5281/zenodo.18757088',
        what='the RICO cumulus field of rte-rrtmgp-cpp, 128 x 128 x 200 cells',
    ),
}


def md5sum(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(chunk), b''):
            h.update(block)

    return h.hexdigest()


def fetch(dataset, force=False):
    """Download the dataset unless it is already here and intact. Returns its path."""
    if os.path.exists(dataset.dest) and not force:
        if md5sum(dataset.dest) == dataset.md5:
            print(f'{dataset.dest} is already here')
            return dataset.dest
        print(f'{dataset.dest} does not match its checksum; fetching it again')

    os.makedirs(os.path.dirname(dataset.dest), exist_ok=True)

    print(f'{dataset.doi}\n  {dataset.url}\n  -> {dataset.dest} '
          f'({dataset.size/1e6:.0f} MB)')

    # Written beside the destination and moved into place at the end, so that an
    # interrupted download never leaves something a case would go on to read.
    partial = dataset.dest + '.part'
    try:
        with urllib.request.urlopen(dataset.url) as response, \
                open(partial, 'wb') as out:
            done = 0
            while block := response.read(1 << 20):
                out.write(block)
                done += len(block)
                if sys.stderr.isatty():
                    print(f'\r  {done/1e6:6.1f} / {dataset.size/1e6:.0f} MB',
                          end='', file=sys.stderr)
        if sys.stderr.isatty():
            print(file=sys.stderr)

        got = md5sum(partial)
        if got != dataset.md5:
            raise SystemExit(f'checksum mismatch: expected {dataset.md5}, got {got}')

        os.replace(partial, dataset.dest)
    finally:
        if os.path.exists(partial):
            os.remove(partial)

    print(f'wrote {dataset.dest}')

    return dataset.dest


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('case', nargs='?', choices=sorted(DATASETS), help='which case')
    p.add_argument('--force', action='store_true',
                   help='download again even if the file is already here')
    args = p.parse_args()

    if args.case is None:
        print('Cases whose input is fetched rather than stored:\n')
        for name, d in sorted(DATASETS.items()):
            print(f'  {name:16s} {d.size/1e6:5.0f} MB  {d.what}\n'
                  f'  {"":16s}        {d.doi}')
        return

    fetch(DATASETS[args.case], force=args.force)


if __name__ == '__main__':
    main()
