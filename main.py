import pandas as pd
import torch.multiprocessing as mp
from Environment import CustomerBehaviorEnv
from Stateandobservation import CustomerState, Observations
from DataReader import PdDataFeeder
from Metric import Metric
from Multi_Actor_Critic import ActorCritic, SharedAdam, Agent, Categorical
from Reward import RewardCalculator

def main():
    # Load the data from the CSV file
    data = pd.read_csv('Processed_data.csv')

    # Create an instance of PdDataFeeder
    data_feeder = PdDataFeeder(data)

    # Initialize the environment
    env = CustomerBehaviorEnv(data_feeder)

    # Define the environment input dimensions and number of actions
    input_dims = env.observation_space.shape
    n_actions = 3  # Manually define the number of actions

    # Global Actor-Critic model
    global_actor_critic = ActorCritic(input_dims, n_actions)
    global_actor_critic.share_memory()  # Share the global model across processes

    # Define the optimizer
    optimizer = SharedAdam(global_actor_critic.parameters(), lr=1e-4)

    # Create shared global episode index
    global_ep_idx = mp.Value('i', 0)

    # Create and start multiple agent processes
    workers = [Agent(global_actor_critic, optimizer, input_dims, n_actions,
                     gamma=0.99, lr=1e-4, name=i, global_ep_idx=global_ep_idx,
                     env=env) for i in range(mp.cpu_count())]

    for worker in workers:
        worker.start()

    for worker in workers:
        worker.join()

if __name__ == "__main__":
    main()
