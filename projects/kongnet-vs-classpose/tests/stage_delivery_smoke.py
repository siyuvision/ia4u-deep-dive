"""Extract the verified package into a fresh directory for relocation testing."""
import argparse
import json
from pathlib import Path, PurePosixPath
import zipfile

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('archive', type=Path)
parser.add_argument('destination', type=Path)
args = parser.parse_args()
destination = args.destination.resolve()
with zipfile.ZipFile(args.archive) as archive:
    for name in archive.namelist():
        path = PurePosixPath(name)
        if path.is_absolute() or '..' in path.parts or ':' in name or '\\' in name:
            raise ValueError('Unsafe archive entry')
    destination.mkdir(parents=True, exist_ok=False)
    archive.extractall(destination)
    manifest = json.loads(archive.read('MANIFEST.json'))
print('Staged', len(manifest['files']), 'files at', destination)
