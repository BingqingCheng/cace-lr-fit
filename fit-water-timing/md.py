import sys
sys.path.append('/global/home/users/dongjinkim/software/cace/')
import os
import numpy as np
import torch
import torch.nn as nn

import cace
from cace.calculators import CACECalculator

import time

from ase import units
from ase.md.langevin import Langevin
from ase.md.npt import NPT
from ase.md.nptberendsen import NPTBerendsen
from ase.md import MDLogger
from ase.io import read, write

r_factor= int(sys.argv[2])
temperature = float(sys.argv[3]) # in K

cace_nnp = torch.load(sys.argv[1])
water_conf = read('../liquid-64.xyz', '0')
init_conf = water_conf.repeat((2,2,r_factor))

calculator = CACECalculator(model_path=cace_nnp, 
                            device='cuda', 
                            energy_key='CACE_energy', 
                            forces_key='CACE_forces',
                            compute_stress=False,
                            atomic_energies={1: -187.6043857100553, 8: -93.80219285502734})

init_conf.set_calculator(calculator)

from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

# Set initial velocities using Maxwell-Boltzmann distribution
MaxwellBoltzmannDistribution(init_conf, temperature * units.kB)

# Define the ensemble
NPTdamping_timescale = 10 * units.fs  # Time constant for NPT dynamics
NVTdamping_timescale = 100 * units.fs  # Time constant for NVT dynamics (NPT includes both)
dyn = NPT(init_conf, timestep=1 * units.fs, temperature_K=temperature,
          ttime=NVTdamping_timescale, pfactor=None, #0.1*NPTdamping_timescale**2,
          externalstress=0.0)

#warm up
steps = 100
dyn.run(steps)

start_time = time.time()

dyn.run(100)

end_time = time.time()

sys_size = 64 *3 * 4* r_factor
elapsed_time = end_time - start_time
md_steps_per_second = steps / elapsed_time
print(f"!!! System_size[atom]: {sys_size} Time_taken[seconds]: {elapsed_time} MD_step/[s]: {md_steps_per_second:.6f}")

