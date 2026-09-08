import os, sys 
sys.setdlopenflags(os.RTLD_NOW | os.RTLD_GLOBAL)

from mpi4py import MPI

import jax
import jax.lax
import jax.numpy as jnp
import numpy as np
jax.config.update("jax_enable_x64", True)
import time
from timeit import timeit
import matplotlib.pyplot as plt
from jax.sharding import PartitionSpec as P
from jax._src import distributed

from dolfinx import mesh, fem
import basix

jax.distributed.initialize()
print(f"Backend: {jax.default_backend()}")
print(f"Global devices: {jax.devices()}\n")
print(f"Local devices: {jax.local_devices()}\n")
cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
print(f"CUDA_VISIBLE_DEVICES = {local_device_ids} ")
local_device_ids = [int(i) for i in cuda_visible_devices.split(",")]
print(distributed.global_state.client)
print(distributed.global_state.service)
print(distributed.global_state.process_id)
# import os
# os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8'
# jax.distributed.initialize() 
