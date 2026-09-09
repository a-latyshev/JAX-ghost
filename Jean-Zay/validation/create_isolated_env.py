"""Developer helper: symlink only JAX/MPI dependencies into a clean venv."""
from pathlib import Path
import subprocess,sys,sysconfig
folder=Path(sys.argv[1])
subprocess.run([sys.executable,'-m','venv','--without-pip',str(folder)],check=True)
site=folder/'lib'/f'python{sys.version_info.major}.{sys.version_info.minor}'/'site-packages'
names={'jax','jaxlib','jax_plugins','jax_cuda12_plugin','jax_cuda13_plugin','nvidia','numpy','numpy.libs','scipy','scipy.libs','ml_dtypes','opt_einsum','mpi4py','typing_extensions.py'}
prefixes=('nvidia_', 'jax-','jaxlib-','jax_cuda12','numpy-','scipy-','ml_dtypes-','opt_einsum-','mpi4py-','typing_extensions-')
for entry in sys.path:
    root=Path(entry)
    if not root.is_dir() or 'site-packages' not in str(root):continue
    for p in root.iterdir():
        if p.name in names or (p.name.endswith('.dist-info') and p.name.startswith(prefixes)):
            dst=site/p.name
            if not dst.exists():dst.symlink_to(p.resolve(),target_is_directory=p.is_dir())
print('Minimal environment:',folder)
