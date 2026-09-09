"""Update release checksums from staged Git contents; stage intended changes first."""
import hashlib
import json
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
names=subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().rstrip('\0').split('\0')
files={}
for name in sorted(names):
    if name=='checksums.json':
        continue
    content=subprocess.check_output(['git','show',':'+name],cwd=ROOT)
    if (ROOT/name).read_bytes()!=content:
        raise ValueError(f'{name}: unstaged changes or checkout newline conversion; stage changes and preserve .gitattributes before sealing a release')
    files[name]=hashlib.sha256(content).hexdigest()
report={'algorithm':'sha256','scope':'Staged release files; this checksum index excludes itself','files':files}
(ROOT/'checksums.json').write_bytes((json.dumps(report,indent=2)+'\n').encode('utf-8'))
print(f'Wrote checksums.json for {len(files)} staged files')
