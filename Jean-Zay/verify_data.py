"""Verify the 12 bundled fixtures and 51 NPZ checksums (standard library)."""
import argparse
import hashlib
import json
from pathlib import Path


def verify(root):
    records=[]
    for size in (4,99):
        for ranks in (range(1,9) if size == 4 else (1,2,4,8)):
            folder=root/f'n{size}/{ranks}ranks'
            meta=json.loads((folder/'metadata.json').read_text())
            if (meta['rank_count']!=ranks or meta['global_dofs']!=(size+1)**3 or
                meta['subdivisions']!=size or meta['dtype']!='float64' or
                meta['settings']['iterations']!=100 or len(meta['fingerprints'])!=ranks):
                raise ValueError(f'Invalid metadata: {folder}')
            if len(list(folder.glob('rank-*.npz')))!=ranks:
                raise ValueError(f'Incorrect rank file count: {folder}')
            total=0
            for rank,wanted in enumerate(meta['fingerprints']):
                p=folder/f'rank-{rank:04d}.npz';h=hashlib.sha256()
                with p.open('rb') as stream:
                    for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
                if h.hexdigest()!=wanted:raise ValueError(f'Checksum mismatch: {p}')
                total+=p.stat().st_size
            records.append(dict(path=str(folder.relative_to(root)),dofs=meta['global_dofs'],ranks=ranks,bytes=total))
    return records


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=Path(__file__).resolve().parent/'data')
    a=p.parse_args();r=verify(a.data)
    print(f"PASS: {len(r)} fixtures, {sum(v['ranks'] for v in r)} rank files, {sum(v['bytes'] for v in r):,} bytes")
