from Functions import *
from neuralflow import energy_model
import matplotlib.pyplot as plt, matplotlib.gridspec as gridspec
import pickle
import sys
import random as rand

EnergyModelParams={
               'peq_model': {'model': 'double_well', 'params': {'xmin': 0.5, 'xmax': 0.0, 'depth': 8}},
               'p0_model':{"model": "single_well", "params": {"miu": 200, "xmin": 0}},
               'D': 0.56,
               'firing_model':{'model': 'rectified_linear', 'params': {'r_slope': 100.0, 'x_thresh': -1}},
               'boundary_mode':'reflecting',
               'verbose':True
               }

em_data_gen = energy_model.EnergyModel(**EnergyModelParams)


# Generate spike data from this model
num_trial = 15
# trials = [rand.uniform(0,1) for i in range(num_trial)]
# trials = np.append(trials,0)
# trials=sorted(trials)
# print(trials)
trial_start = np.zeros(num_trial) #[trials[i] for i in range(num_trial)]
trial_end = np.ones(num_trial) #[trials[i+1] for i in range(num_trial)]
data_ISI, time_bins, diff_traj = em_data_gen.generate_data(deltaT=0.00001,trial_start=trial_start,trial_end=trial_end)


EnergyModelParams={
                'peq_model':{"model": "uniform", "params": {}},
                'p0_model':{"model": "single_well", "params": {"miu": 200, "xmin": 0}},
                'D': 0.56,
                'firing_model':{'model': 'rectified_linear', 'params': {'r_slope': 100.0, 'x_thresh': -1}},
                'boundary_mode':'reflecting',
                'verbose':True
                # 'boundary_mode':'reflecting',
                # 'p0_model':{"model": "single_well", "params": {"miu": 200, "xmin": 0}},
                # 'firing_model':[{"model": "linear", "params": {"r_slope": 50, "r_bias": 60}}],
                # 'verbose':True
               }
em_fitting = energy_model.EnergyModel(**EnergyModelParams)

# Cell 3

learning_rate = 0.05
number_of_iterations = 10

options={}
options['data']={'dataTR':data_ISI}#,'dataCV':data_ISI}
options['optimization']= {'gamma':{'params_to_opt':['F'],'alpha':learning_rate,'beta1':0.9,'beta2':0.999,'epsilon':10**(-8)},'max_iteration':number_of_iterations}
em_fitting = energy_model.EnergyModel(**EnergyModelParams)
em_fitting.fit(data = options['data'], optimizer = 'ADAM',optimization = options['optimization'],save={'path':"/home/db8682/langevin_rotation/job_data",'stride':1})
