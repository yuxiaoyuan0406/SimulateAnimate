import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from analyze.signal import Signal, SquareWave
from analyze.signal import UnitStep

matplotlib.use('TkAgg')


def SquareWaveIdealTest(timeout=1., dt=5e-7, freq=62.5 * 1e3, amp=.5, bias=.5):
    """Do square wave test.

    Args:
        timeout (float, optional): Total time. Defaults to 1..
        dt (float, optional): Time gap. Defaults to 5e-7.
        freq (float, optional): Frequancy of square wave. Defaults to 62.5*1e3.
        amp (float, optional): The amplitude of squarewave. Defaults to .5.
        bias (float, optional): The bias of squarewave. Defaults to .5.
    """
    t = np.linspace(0, timeout, int(timeout / dt), endpoint=False)

    ax_time, ax_power, ax_phase = None, None, None

    ideal_square = SquareWave(t=t,
                              freq=freq,
                              amp=amp,
                              bias=bias,
                              ideal=True,
                              label='ideal')

    ax_time = ideal_square.plot_time_domain(ax=ax_time)
    ax_power, ax_phase = ideal_square.plot_freq_domain(ax_power=ax_power,
                                                       ax_phase=ax_phase)

    # nideal_square = chop_gen(t, 32)
    nideal_square = SquareWave(t=t,
                               freq=freq,
                               amp=amp,
                               bias=bias,
                               ideal=False,
                               label='non-ideal')

    ax_time = nideal_square.plot_time_domain(ax=ax_time)
    ax_power, ax_phase = nideal_square.plot_freq_domain(ax_power=ax_power,
                                                        ax_phase=ax_phase)


def UnitStepSignalTest(timeout=1., dt=5e-7, freq=62.5 * 1e3, amp=.5, t0=0.1):
    t = np.linspace(0, timeout, int(timeout / dt), endpoint=False)
    ax_time, ax_power, ax_phase = None, None, None

    step = UnitStep(t, t0, amp, label='unit step')

    ax_time = step.plot_time_domain(ax=ax_time)
    ax_power, ax_phase = step.plot_freq_domain(ax_power=ax_power,
                                               ax_phase=ax_phase)


if __name__ == '__main__':
    # SquareWaveIdealTest()
    UnitStepSignalTest(t0=0)

    plt.show(block=False)
    input('Enter to exit...')
    plt.close('all')
