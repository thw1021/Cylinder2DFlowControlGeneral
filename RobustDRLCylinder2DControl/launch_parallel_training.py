import argparse
import os
import sys
import csv
import socket
import numpy as np

from tensorforce.agents import Agent
from tensorforce.execution import ParallelRunner

from simulation_base.env import resume_env, nb_actuations, simulation_duration, dt
from RemoteEnvironmentClient import RemoteEnvironmentClient


ap = argparse.ArgumentParser()
ap.add_argument("-n", "--number-servers", required=True, help="number of servers to spawn", type=int)
ap.add_argument("-p", "--ports-start", required=True, help="the start of the range of ports to use", type=int)
ap.add_argument("-t", "--host", default="None", help="the host; default is local host; string either internet domain or IPv4", type=str)

args = vars(ap.parse_args())

number_servers = args["number_servers"]
ports_start = args["ports_start"]
host = args["host"]

if host == 'None':
    host = socket.gethostname()

dump = 100
example_environment = resume_env(plot=False, step=100, dump=dump)

use_best_model = True

environments = []
for crrt_simu in range(number_servers):
    environments.append(RemoteEnvironmentClient(
        example_environment, verbose=0, port=ports_start + crrt_simu, host=host,
        timing_print=(crrt_simu == 0)
    ))

if use_best_model:
    evaluation_environment = environments.pop()
else:
    evaluation_environment = None

network = [dict(type='dense', size=512), dict(type='dense', size=512)]

learning_rate = 1e-3
decaying_lr = learning_rate

# parameter_decay_step = 1000
# parameter_decay_rate = 0.5
#
# decaying_lr = dict(
#     type='decaying', dtype='float', unit="episodes",
#     decay="exponential", initial_value=learning_rate,
#     decay_steps=parameter_decay_step, decay_rate=parameter_decay_rate
# )

agent = Agent.create(
    # Agent + Environment
    agent='ppo', environment=example_environment, max_episode_timesteps=nb_actuations,
    # TODO: nb_actuations could be specified by Environment.max_episode_timesteps() if it makes sense...
    # Network
    network=network,
    # Optimization
    batch_size=70, learning_rate=decaying_lr, subsampling_fraction=0.2, optimization_steps=25,
    # Reward estimation
    likelihood_ratio_clipping=0.2,
    # TODO: gae_lambda=0.97 doesn't currently exist
    # Critic
    critic_network=network,
    critic_optimizer=dict(
        type='multi_step', num_steps=5,
        optimizer=dict(type='adam', learning_rate=1e-3)
    ),
    # Regularization
    entropy_regularization=0.01,
    # TensorFlow etc
    parallel_interactions=number_servers,
    saver=dict(directory=os.path.join(os.getcwd(), 'saver_data'), frequency=72000),  # the high value of the seconds parameter here is so that no erase of best_model
    recorder=dict(directory=os.path.join(os.getcwd(), 'recorder_data'), frequency=20)
)

agent.initialize()
cwd = os.getcwd()

# actually, called automatically and crashes if calling twice.
# if (os.path.exists(cwd + "/saver_data/") and os.listdir(cwd + "/saver_data/")):
#     print("\n*************  Note that training will load existed model *************\n")
#     agent.initialize()

runner = ParallelRunner(
    agent=agent, environments=environments, evaluation_environment=evaluation_environment,
    save_best_agent=use_best_model
)

evaluation_folder = "env_" + str(number_servers - 1)
sys.path.append(cwd + evaluation_folder)
# out_drag_file = open("avg_drag.txt", "w")

def evaluation_callback_1(r):
    if(not os.path.exists(evaluation_folder + "/saved_models/output.csv")):
        print("no output.csv file, check path\n")
        sys.exit()
    else:
        with open(evaluation_folder + "/saved_models/output.csv", 'r') as csvfile:
            data = csv.reader(csvfile, delimiter = ';')
            for row in data:
                lastrow = row
            avg_drag = float(lastrow[1])

#     drag = '%10.6f\n' % (avg_drag)
#     out_drag_file.write(drag)

    return avg_drag

half_epoch = int(simulation_duration/dt/dump/2) * (-1)

def evaluation_callback_2(r):
    if(not os.path.exists(evaluation_folder + "/saved_models/debug.csv")):
        print("no debug.csv file, check path\n")
        sys.exit()
    else:
        debug_data = np.genfromtxt(evaluation_folder + "/saved_models/debug.csv", delimiter=";")
        debug_data = debug_data[1:,1:]
        avg_data = np.average(debug_data[half_epoch:], axis=0)
        avg_drag = avg_data[3]

#     drag = '%10.6f\n' % (avg_drag)
#     out_drag_file.write(drag)

    if np.isnan(avg_data).any() or np.isinf(avg_data).any():
        print("------- hit NaN in callback function -------")
        return(-100)

    return avg_drag

runner.run(
    num_episodes=2100, max_episode_timesteps=nb_actuations, sync_episodes=False,
    evaluation_callback=evaluation_callback_2
)
# out_drag_file.close()
runner.close()
