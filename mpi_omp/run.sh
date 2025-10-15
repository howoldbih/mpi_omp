#!/bin/bash

rm -rf logs
mkdir logs

REPEATS=10

PROCESSES=(1 2 4 8 16)

MPI_EXECUTABLE="combined_mpi_taylor"
OMP_EXECUTABLE="combined_omp_taylor"

MPI_SOURCE="${MPI_EXECUTABLE}.c"
OMP_SOURCE="${OMP_EXECUTABLE}.c"

export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_SCHEDULE=dynamic,64

module load mpi/openmpi-x86_64

echo "Компиляция MPI версии ($MPI_SOURCE)..."
mpicc -o $MPI_EXECUTABLE $MPI_SOURCE -lm -O3 

echo "Компиляция OpenMP версии ($OMP_SOURCE)..."
gcc -fopenmp -o $OMP_EXECUTABLE $OMP_SOURCE -lm -O3

run_mpi_jobs() {
    for procs in "${PROCESSES[@]}"; do
        for ((i = 1; i <= REPEATS; i++)); do
            JOB_NAME="MPI_${procs}_procs_run${i}"

            if [ "$procs" -le 2 ]; then
                WALLTIME="00:05"
                MEM="512MB"
            elif [ "$procs" -le 8 ]; then
                WALLTIME="00:10"
                MEM="1GB"
            else
                WALLTIME="00:20"
                MEM="2GB"
            fi

            bsub <<EOF
#!/bin/bash
#BSUB -J $JOB_NAME
#BSUB -W $WALLTIME
#BSUB -n $procs
#BSUB -R "span[ptile=$procs]"
#BSUB -o logs/output_${JOB_NAME}_%J.out
#BSUB -e logs/error_${JOB_NAME}_%J.err
#BSUB -M $MEM

ulimit -l unlimited
module load mpi/openmpi-x86_64
mpirun -np $procs --mca pml ^cm,ucx --mca ptm ^posix --mca btl tcp,self --bind-to core --map-by core ./$MPI_EXECUTABLE
EOF

            sleep 0.2
        done
    done
}

run_omp_jobs() {
    for threads in "${PROCESSES[@]}"; do
        for ((i = 1; i <= REPEATS; i++)); do
            JOB_NAME="OMP_${threads}_threads_run${i}"

            if [ "$threads" -le 2 ]; then
                WALLTIME="01:00"
                MEM="512MB"
            elif [ "$threads" -le 8 ]; then
                WALLTIME="02:10"
                MEM="1GB"
            else
                WALLTIME="03:20"
                MEM="2GB"
            fi

            bsub <<EOF
#!/bin/bash
#BSUB -J $JOB_NAME
#BSUB -W $WALLTIME
#BSUB -n $threads
#BSUB -R "span[ptile=$threads]"
#BSUB -o logs/output_${JOB_NAME}_%J.out
#BSUB -e logs/error_${JOB_NAME}_%J.err
#BSUB -M $MEM

module load mpi/openmpi-x86_64
export OMP_NUM_THREADS=$threads
./$OMP_EXECUTABLE
EOF

            sleep 0.2
        done
    done
}

run_mpi_jobs
run_omp_jobs