"""Verify the ALFWorld release inventory and file hashes on CPU."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def verify():
    files = json.loads((ROOT / 'checksums.json').read_text(encoding='utf-8'))['files']
    if (ROOT / '.git').exists():
        tracked = set(subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0')) - {''}
        if set(files) != tracked - {'checksums.json'}:
            raise ValueError('Checksum inventory differs from tracked release files')
    for name, expected in files.items():
        path = (ROOT / name).resolve()
        if ROOT.resolve() not in path.parents:
            raise ValueError('Checksum path escapes the repository: ' + name)
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('Missing or changed release file: ' + name)
    return {'passed': True, 'checked_files': len(files), 'algorithm': 'sha256',
            'verification': 'saved ALFWorld files; no model inference or training'}


if __name__ == '__main__':
    print(json.dumps(verify(), indent=2))
