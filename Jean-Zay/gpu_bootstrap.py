"""Select GPU visibility before CUDA-aware MPI can initialize the CUDA driver.

Only standard-library imports are permitted here. MPI later verifies the rank
and physical device assignments; launcher environment variables are hints only.
"""
import os


def bootstrap():
    for key in ('OMPI_COMM_WORLD_LOCAL_RANK', 'MPI_LOCALRANKID', 'SLURM_LOCALID'):
        if key not in os.environ:
            continue
        try:
            rank = int(os.environ[key])
            if rank < 0:
                raise ValueError('Negative local rank')
            visible = os.environ.get('CUDA_VISIBLE_DEVICES')
            if visible is None:
                os.environ['CUDA_VISIBLE_DEVICES'] = str(rank)
            else:
                ids = [s.strip() for s in visible.split(',') if s.strip()]
                if not ids or ids == ['-1']:
                    raise ValueError('No visible GPU; request GPUs from the scheduler')
                if len(ids) > 1:
                    if rank >= len(ids):
                        raise ValueError('More local ranks than visible GPUs')
                    os.environ['CUDA_VISIBLE_DEVICES'] = ids[rank]
            return rank, None
        except ValueError as exc:
            # Defer failure until MPI can report it collectively.
            return None, f'GPU bootstrap ({key}): {exc}'
    return None, None  # Unknown launcher: retain the MPI-based selection path.
