import numpy as np
import simpy
from tqdm import tqdm
import matplotlib.pyplot as plt
from simu import Planet
from simu import MultiPlanetSystem
from analyze import *

if __name__ == "__main__":
    day = 24 * 60 * 60.  # 24h * 60min * 60sec
    year = 1000. * day

    dt = day / 50
    runtime = year

    env = simpy.Environment(0)

    sun = Planet(env, 1.989e30, runtime=runtime, dt=dt, name='Sun')
    earth = Planet(env,
                   5.972e24,
                   initial_position=np.array([1.4959e11, 0]),
                   initial_velocity=np.array([0, 3e4]),
                   runtime=runtime,
                   dt=dt,
                   name='Earth')

    solar = MultiPlanetSystem(env,
                              planets=[earth, sun],
                              runtime=runtime,
                              dt=dt)

    with tqdm(total=int(runtime / dt), desc='Solar simulation') as pbar:
        while env.now < runtime:
            env.run(until=env.now + dt)
            pbar.update(1)

    data = np.array(solar.history)
    np.save('solar', data)

    earth_pos = np.array(earth.simulation_data['position']).transpose()
    sun_pos = np.array(sun.simulation_data['position']).transpose()

    fig, ax = plt.subplots()
    ax.plot(earth_pos[0], earth_pos[1], label='earth')
    ax.plot(sun_pos[0], sun_pos[1], label='sun')

    ax.legend(loc='upper right')
    ax.set_aspect('equal')
    ax.grid(True)

    plt.show(block=True)
