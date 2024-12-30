import os
import pickle
from vidur.config import SimulationConfig
from vidur.simulator import Simulator
from vidur.utils.random import set_seeds
from vidur.utils.model_persistence import get_model_path,PREDICTOR_OUTPUT_DIR
from vidur.scheduler.replica_scheduler.base_replica_scheduler import BaseReplicaScheduler

def main() -> None:
    config: SimulationConfig = SimulationConfig.create_from_cli_args()

    set_seeds(config.seed)

    simulator = Simulator(config)
    simulator.run()
    scheduler:BaseReplicaScheduler = None   
    if not os.path.exists(PREDICTOR_OUTPUT_DIR):
        os.mkdir(PREDICTOR_OUTPUT_DIR)
    for id,scheduler in simulator.scheduler._replica_schedulers.items():
        model_name = scheduler._replica_config.model_name
        tp_num = scheduler._replica_config.tensor_parallel_size
        mem_margin = scheduler._replica_config.memory_margin_fraction
        model_path = get_model_path(model_name,tp_num,mem_margin)
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        with open(model_path, 'wb') as file:
            pickle.dump(scheduler._replica_stage_schedulers[0]._execution_time_predictor, file)



if __name__ == "__main__":
    main()
