import numpy as np
import simpy

g = 9.80665

class Module:
    def __init__(
        self,
        env: simpy.Environment,
        mass = 1.0,
        length = 1.0,
        center = np.array([0,0], dtype=np.float64),
        init_angle = 0,
        init_velo = 0,
        runtime = 1.0,
        dt = 1/30,
    ) -> None:
        self.env = env
        self.m = mass
        self.g = np.array([0, -g])
        self.l = length
        self.center = center
        self.r = np.array([np.cos(init_angle), np.sin(init_angle)]) * length
        init_posi = center + self.r
        init_velo = np.array([-np.sin(init_angle), np.cos(init_angle)]) * init_velo
        self.state = np.array([init_posi, init_velo])
#       self.posi = init_state[0]
#       self.velo = init_state[1]
        self.runtime = runtime
        self.dt = dt
        
        self.simulation_data = {
            'time': [],
            'position': [],
            'velocity': [],
        }
        
        self.env.process(self.run())
        
    def state_equation(self, state, t):
        pos, v = state
        r = pos - self.center
        a = self.g - (self.g @ r) * r / np.linalg.norm(r)
        return np.array([v,a])
    
    def update(self):
        t = self.env.now
        current_state = self.state
        k1 = self.state_equation(current_state, t)
        k2 = self.state_equation(current_state + k1 * self.dt/2, t + self.dt/2)
        k3 = self.state_equation(current_state + k2 * self.dt/2, t + self.dt/2)
        k4 = self.state_equation(current_state + k3 * self.dt, t + self.dt)
        
        k = (k1 + 2*k2 + 2*k3 + k4) / 6
        self.state = current_state + k * self.dt
        
    def run(self):
        while self.env.now < self.runtime:
            self.simulation_data['time'].append(self.env.now)
            self.simulation_data['position'].append(self.state[0])
            self.simulation_data['velocity'].append(self.state[1])
            self.update()
            yield self.env.timeout(self.dt)
            
