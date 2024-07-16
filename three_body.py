from turtle import color
import numpy as np
import simpy
from tqdm import tqdm
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from simu import Planet
from simu import MultiPlanetSystem
from anal import *

from simu import module

module.G = 1

FPS = 60

R = 0.5

current_time = 0

def simu_data_gen():
    # dt = 1 / FPS / 15
    dt = 1e-4
    runtime = 2e8

    env = simpy.Environment(0)

    init_pos = [R * np.array(
        [np.cos(i * 2 * np.pi / 3),
        np.sin(i * 2 * np.pi / 3)]) for i in range(3)]
    init_vel = [np.array(
        [-np.sin(i * 2 * np.pi / 3),
        np.cos(i * 2 * np.pi / 3)]) + np.random.uniform(-1,1, size=(2,)) for i in range(3)]
    planets = [
        Planet(env,
               np.random.uniform(0.9, 1.1),
               initial_position=init_pos[i],
               initial_velocity=init_vel[i],
               runtime=runtime,
               dt=dt) for i in range(3)
    ]

    s = MultiPlanetSystem(env, planets, runtime=runtime, dt=dt)

    while env.now < runtime:
        env.run(until=env.now + 1 / FPS)
        global current_time
        current_time = env.now
        yield s.state


x_min = y_min = -R
x_max = y_max = R

if __name__ == "__main__":
    fig, ax = plt.subplots()
    color_list = ['red', 'green', 'blue']
    anima_points = [ax.plot([], [], color=c, marker='o')[0] for c in color_list]
    time_text = ax.text(0.02, 1.02, '', transform=ax.transAxes)
    # mass_text = [ax.text(0,0, f'm={}')]
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal')
    ax.grid(True)

    def anima_init():
        for anima_point in anima_points:
            anima_point.set_data([], [])
        return anima_points

    def anima_update(data):
        global x_min, x_max, y_min, y_max
        global current_time
        x = [x_min, x_max]
        y = [y_min, y_max]

        for i in range(len(data)):
            p_state = data[i]
            point = anima_points[i]

            pos = p_state[0]
            point.set_data([pos[0]], [pos[1]])
            x.append(pos[0])
            y.append(pos[1])

        x_min, x_max = min(x), max(x)
        y_min, y_max = min(y), max(y)

        time_text.set_text(f't={current_time:.3f}s')

        ax.set_xlim(x_min * 1.2, x_max * 1.2)
        ax.set_ylim(y_min * 1.2, y_max * 1.2)
        return anima_points

    ani = FuncAnimation(fig,
                        anima_update,
                        frames=simu_data_gen,
                        init_func=anima_init,
                        blit=False,
                        interval=1000 / FPS,
                        save_count=int(60*FPS),
                        )

    plt.show()

    # ani.save(f'data/three-body-r={R}.mp4', writer='ffmpeg', dpi=300, fps=FPS)
