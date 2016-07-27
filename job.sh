#!/bin/bash
#PBS -q regular
#PBS -l nodes=16:ppn=20
#PBS -l walltime=12:00:00
#PBS -T flush_cache

cd $PBS_O_WORKDIR

module load openmpi/1.8.3/gnu/4.9.2
module load mkl/11.1

mpirun --map-by ppr:5:socket ./32_master.x -mfn_converged_reason -memory_view > 320p_32half_tmax100_5pprpersocket_master.dat
