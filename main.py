import numpy as np
import simpy
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from modules import Pendulum

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

    with tqdm(total=int(runtime/dt), desc='Running simulation') as pbar:
        while env.now < runtime:
            env.run(until=env.now + dt)
            pbar.update(1)
    
    t = system.simulation_data['time']
    pos = np.array(system.simulation_data['position'])
    velo = np.array(system.simulation_data['velocity'])
    
    pos_x, pos_y = pos.transpose()
    
    fig, ax = plt.subplots()
    ax.set_xlim([np.min(pos_x) * 1.2, np.max(pos_x) * 1.2])
    ax.set_ylim([np.min(pos_y) * 1.2, np.max(pos_y) * 1.2])
    line, =ax.plot([], [], 'o-', lw=2)
    
    _ = []
    for i in range(len(pos)):
        if i % k == 0:
            _.append(pos[i])
    pos = _
    
    def anim_init():
        line.set_data([], [])
        return line,

    def anim_update(frame):
        start=system.center
        end = pos[frame]
        line.set_data([start[0], end[0]], [start[1], end[1]])
        return line,
    
    ani = FuncAnimation(fig, anim_update, frames=len(pos), init_func=anim_init, blit=True, interval=1000/fr)
    
    plt.show()
    ani.save('simulation.mp4', writer='ffmpeg', fps=fr)
   