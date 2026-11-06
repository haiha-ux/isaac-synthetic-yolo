set -e
export OMP_NUM_THREADS=4
g() { nice ~/isaacsim/python.sh gen_data.py --mode $1 --n $2 --seed $3 --out data/$4 > logs_gen_$4.log 2>&1; }
g rand 2000 0 rand_train
g rand 300 1 rand_val
g shift 300 2 shift_test
g fixed 2000 3 fixed_train
g fixed 300 4 fixed_val
echo ALLDONE > gen_all.done
