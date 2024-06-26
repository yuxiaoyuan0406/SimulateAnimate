import numpy as np
import simpy
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from simu import Pendulum

if __name__ == "__main__":
    fr = 30
    k = 100

    runtime = 15
    dt = 1/(fr * k)
    env = simpy.Environment(0)
    system = Pendulum(env=env,
                      mass=1,
                      length=1,
                      center=np.array([0, 0], dtype=np.float64),
                      init_angle=0,
                      init_speed=0,
                      runtime=runtime,
                      dt=dt)

    with tqdm(total=int(runtime / dt), desc='Running simulation') as pbar:
        while env.now < runtime:
            env.run(until=env.now + dt)
            pbar.update(1)

    t = system.simulation_data['time']
    pos = np.array(system.simulation_data['position'])
    velo = np.array(system.simulation_data['velocity'])

    pos_x, pos_y = pos.transpose()

    fig, ax = plt.subplots()
    ax.grid(True)
    x_min = np.min(pos_x)
    x_max = np.max(pos_x)
    x_diff = x_max - x_min
    ax.set_xlim([x_min - x_diff * 0.1, x_max + x_diff * 0.1])
    y_min = np.min(pos_y)
    y_max = np.max(pos_y)
    y_diff = y_max - y_min
    ax.set_ylim([y_min - y_diff * 0.1, y_max + y_diff * 0.1])
    ax.set_aspect('equal')

    line, = ax.plot([], [], 'o-', lw=2)

    _ = []
    for i in range(len(pos)):
        if i % k == 0:
            _.append(pos[i])
    pos = _

    def anim_init():
        line.set_data([], [])
        return line,

    def anim_update(frame):
        start = system.center
        end = pos[frame]
        line.set_data([start[0], end[0]], [start[1], end[1]])
        return line,

    ani = FuncAnimation(fig,
                        anim_update,
                        frames=len(pos),
                        init_func=anim_init,
                        blit=True,
                        interval=1000 / fr)

    plt.show()
    # ani.save('simulation.mp4', writer='ffmpeg', fps=fr)
