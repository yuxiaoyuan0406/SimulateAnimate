import numpy as np
import simpy
from tqdm import tqdm
import matplotlib.pyplot as plt
from simu import Pendulum
from analyze import *

if __name__ == "__main__":
    fr = 10
    k = 100

    runtime = 50
    dt = 1 / (fr * k)

    def simulation(mass, length, init_angle, init_speed, label):
        env = simpy.Environment(0)
        system = Pendulum(env=env,
                          mass=mass,
                          length=length,
                          center=np.array([0, 0], dtype=np.float64),
                          init_angle=init_angle,
                          init_speed=init_speed,
                          runtime=runtime,
                          dt=dt)

        # with tqdm(total=int(runtime / dt), desc=f'Running simulation {label}') as pbar:
        # while env.now < runtime:
        # env.run(until=env.now + dt)
        # pbar.update(1)

        env.run(until=runtime)

        t = system.simulation_data['time']
        pos = np.array(system.simulation_data['position'])
        # velo = np.array(system.simulation_data['velocity'])

        pos_x, pos_y = pos.transpose()
        return Signal(pos_x, t=t, label=label)

    # x = []
    # for i in range(4):
    # init_angle = - 0.5 * i
    # x.append(simulation(mass=1, length=0.1, init_angle=init_angle, init_speed=0, label=f'init angle {init_angle}'))

    # ax_time, ax_power, ax_phase = None, None, None
    # for _ in x:
    # ax_time = _.plot_time_domain(ax=ax_time)
    # ax_power, ax_phase = _.plot_freq_domain(ax_power=ax_power, ax_phase=ax_phase)
    # print(f'{_.label} has a dominant frequency of {_.dominant_freq()}')

    init_angle = np.linspace(-np.pi / 2 * 0.99, 0, 1000)
    dominant_freq = []
    for i in tqdm(init_angle):
        dominant_freq.append(
            simulation(mass=1,
                       length=0.1,
                       init_angle=i,
                       init_speed=0,
                       label=f'init angle {i}').dominant_freq())

    plt.plot(np.degrees(init_angle), dominant_freq)

    plt.show()

    data = np.array([init_angle, dominant_freq])
    data = data.transpose()
    np.savetxt('init_angle_dominant_freq.txt', data)
