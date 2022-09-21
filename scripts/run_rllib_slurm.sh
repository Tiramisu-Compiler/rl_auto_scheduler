#!/bin/bash
#SBATCH -p research
#SBATCH --nodes=2
#SBATCH --exclusive
#SBATCH --tasks-per-node 1
#SBATCH --cpus-per-task=28
#SBATCH -t 7-0:00:00
#SBATCH -o outputs/train.out
#SBATCH -e outputs/train.err

bash scripts/env.sh

export RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE=1
export RAY_ALLOW_SLOW_STORAGE=1

# Getting the node names
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
#nodes_array=dn[096,102-104]
nodes_array=($nodes)

head_node=${nodes_array[0]}
echo "head node is $head_node"
head_node_ip=$(srun --ntasks=1 -w "$head_node" hostname --ip-address)
echo "IP address is $head_node_ip"

# if we detect a space character in the head node IP, we'll
# convert it to an ipv4 address. This step is optional.
if [[ "$head_node_ip" == *" "* ]]; then
IFS=' ' read -ra ADDR <<<"$head_node_ip"
if [[ ${#ADDR[0]} -gt 16 ]]; then
  head_node_ip=${ADDR[1]}
else
  head_node_ip=${ADDR[0]}
fi
echo "IPV6 address detected. We split the IPV4 address as $head_node_ip"
fi
# __doc_head_address_end__

# __doc_head_ray_start__
port=6379
ip_head=$head_node_ip:$port
export ip_head
echo "IP Head: $ip_head"

#export RAY_ALLOW_SLOW_STORAGE=1

echo "Starting HEAD at $head_node"
srun --nodes=1 -N1 --ntasks=1 -w "$head_node" \
    ray start --head --object-store-memory=1000000000 --node-ip-address="$head_node_ip" --port=$port \
    --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus 1 --block &
# __doc_head_ray_end__


# number of nodes other than the head node
worker_num=$((SLURM_JOB_NUM_NODES - 1))
worker_per_code=10

for ((i = 1; i <= worker_num; i++)); do
  for ((w = 1; w <= worker_per_code; w++)); do
      echo "running worker $w on node $i"
      node_i=${nodes_array[$i]}
      # echo "tarting WORKER $i at $node_i"
      srun --nodes=1 -N1 --ntasks=1 -w "$node_i" \
          ray start --address "$ip_head" \
          --num-cpus "${SLURM_CPUS_PER_TASK}" --block > /dev/null &
      # sleep 10
  done
done
# __doc_worker_ray_end__

# __doc_script_start__

echo $(expr $worker_per_code \* $worker_num)
python train_ppo.py ray.num-workers=$(expr $worker_per_code \* $worker_num) ray.env_type=model