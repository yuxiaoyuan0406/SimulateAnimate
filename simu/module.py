import numpy as np
import simpy

earth_gravity = 9.80665


class Pendulum:

    def __init__(
        self,
        env: simpy.Environment,
        mass: float = 1.0,
        length: float = 1.0,
        center=np.array([0, 0], dtype=np.float64),
        init_angle: float = 0.,
        init_speed: float = 0.,
        runtime: float = 1.0,
        dt: float = 1 / 30,
    ) -> None:
        self.env = env
        self.m = mass
        self.g = np.array([0, -earth_gravity])
        self.l = length
        self.center = center
        self.r = np.array([np.cos(init_angle), np.sin(init_angle)]) * length
        init_posi = center + self.r
        init_velo = np.array([-np.sin(init_angle),
                              np.cos(init_angle)]) * init_speed

        ## System State
        ## angle, angular velocity
        self.angular_state = np.array([init_angle, init_speed / length],
                                      dtype=np.float64)
        self.linear_state = np.array([init_posi, init_velo], dtype=np.float64)

        self.runtime = runtime
        self.dt = dt

        self.simulation_data = {
            'time': [],
            'angle': [],
            'a_velocity': [],
            'position': [],
            'velocity': [],
        }

        self.env.process(self.run())

    def _angle_to_linear(self, angular_state):
        theta, omega = angular_state
        r = np.array([np.cos(theta), np.sin(theta)])
        r_tangent = np.dot(np.array([[0, -1], [1, 0]]), r)
        v = r_tangent * omega * self.l
        r *= self.l
        return r, r_tangent, v

    def state_equation(self, state, t):
        theta, omega = state
        r, r_tangent, v = self._angle_to_linear(state)
        a_tangent = self.g @ r_tangent * r_tangent
        if r_tangent[0] > 1e-9:
            alpha = a_tangent[0] / (r_tangent[0] * self.l)
        else:
            alpha = a_tangent[1] / (r_tangent[1] * self.l)
        return np.array([omega, alpha])

    def update(self):
        t = self.env.now
        current_state = self.angular_state
        k1 = self.state_equation(current_state, t)
        k2 = self.state_equation(current_state + k1 * self.dt / 2,
                                 t + self.dt / 2)
        k3 = self.state_equation(current_state + k2 * self.dt / 2,
                                 t + self.dt / 2)
        k4 = self.state_equation(current_state + k3 * self.dt, t + self.dt)

        k = (k1 + 2 * k2 + 2 * k3 + k4) / 6
        theta, omega = current_state + k * self.dt
        self.angular_state = np.array([theta, omega])
        r, r_tangent, v = self._angle_to_linear([theta, omega])
        self.linear_state = np.array([r + self.center, v])

    def run(self):
        while self.env.now < self.runtime:
            self.simulation_data['time'].append(self.env.now)
            self.simulation_data['angle'].append(self.angular_state[0])
            self.simulation_data['a_velocity'].append(self.angular_state[1])
            self.simulation_data['position'].append(self.linear_state[0])
            self.simulation_data['velocity'].append(self.linear_state[1])
            self.update()
            yield self.env.timeout(self.dt)

G = 6.67430e-11

class Planet:
    __planet_default_name_counter__ = 0

    def __init__(
        self,
        env: simpy.Environment,
        mass: float,
        initial_position = np.array([0,0], dtype=np.float64),
        initial_velocity = np.array([0,0], dtype=np.float64),
        runtime: float = 1,
        dt: float = 1e-3,
        name: str = '',
    ) -> None:
        self.env = env
        self.mass = mass
        self.state = np.array([initial_position, initial_velocity])
        self.runtime = runtime
        self.dt = dt

        self.simulation_data = {
            'time': [],
            'position': [],
            'velocity': [],
        }
        if name == '':
            name = f'Planet{Planet.__planet_default_name_counter__}'
            Planet.__planet_default_name_counter__ += 1
        self.name = name

    # def state_equation(self, state, t):
        # pos, vel = state
        # grav_acce = np.zeros(shape=self.state[1].shape, dtype=np.float64)

    def update(self, state):
        self.save_state()
        self.state = state
    
    def save_state(self):
        self.simulation_data['time'].append(self.env.now)
        self.simulation_data['position'].append(self.state[0])
        self.simulation_data['velocity'].append(self.state[1])

def gravity(target: Planet, source: Planet):
    r = target.state[0] - source.state[0]
    return G * target.mass * source.mass / (np.linalg.norm(r) ** 3) * r

class MultiPlanetSystem:
    def __init__(
        self,
        env: simpy.Environment,
        planets: list[Planet],
        runtime: float = 1,
        dt: float = 1e-3,
    ) -> None:
        self.env = env
        assert len(planets) != 0, "Initializing a Multi-planet system with no planets."
        self.planets = planets
        self.runtime = runtime
        self.dt = dt
        self.state = []
        for planet in planets:
            self.state.append(planet.state)
        self.state = np.array(self.state)
        
        self.history = []

        self.env.process(self.run())

    def state_equation(self, state, t):
        # Assume state is a array of shape (len(self.planets), 2).
        # For example, if 3 plantes are running in a plain surface,
        # then the state shape should be (3,2).
        # Each state in state list should countain a position and velocity.
        # And the mass of each planet is given by self.planet
        # velo = []
        # acce = []
        ret = []
        mass = [planet.mass for planet in self.planets]
        assert len(mass) == len(state), f"Input state({len(state)}) has a different dimention with planets({len(mass)})."
        for i in range(len(state)):
            s = state[i]
            # velo.append(s[1])
            a = 0
            for j in range(len(mass)):
                if i != j:
                    target = state[j]
                    r = target[0] - s[0]
                    a += G * mass[j] / (np.linalg.norm(r) ** 3) * r
            # acce.append(a)
            ret.append(np.array([s[1], a], dtype=np.float64))
        # ret = np.array([velo, acce], dtype=np.float64).transpose()
        ret = np.array(ret)
        return ret


    def update(self):
        t = self.env.now
        current_state = self.state
        k1 = self.state_equation(current_state, t)
        k2 = self.state_equation(current_state + k1 * self.dt / 2,
                                 t + self.dt / 2)
        k3 = self.state_equation(current_state + k2 * self.dt / 2,
                                 t + self.dt / 2)
        k4 = self.state_equation(current_state + k3 * self.dt, t + self.dt)

        k = (k1 + 2 * k2 + 2 * k3 + k4) / 6
        state = current_state + k * self.dt
        self.state = state
        for i in range(len(self.planets)):
            planet = self.planets[i]
            planet.update(state[i])

    def run(self):
        while self.env.now < self.runtime:
            self.history.append(self.state)
            self.update()
            yield self.env.timeout(self.dt)
