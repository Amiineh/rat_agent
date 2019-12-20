from baselines.common import plot_util as pu
import matplotlib.pyplot as plt
import numpy as np

for i in range(5):
    results = pu.load_results('/Users/amineh.ahm/Desktop/Mice/code/rat_exp/acktr_dm 2/train_' + str(i))
    r = results[0]

    # plt.plot(np.cumsum(r.monitor.l), r.monitor.r)
    # plt.show()
    plt.plot(np.cumsum(r.monitor.l), pu.smooth(r.monitor.r, radius=10))
    plt.show()
    # plt.plot(r.progress.total_timesteps, r.progress.eprewmean)
    # plt.show()
    # f, axarr = pu.plot_results(results, average_group=True, split_fn=lambda _: '')  # pu.plot_results(results)
    # f.show()
    # axarr[0, 0].show()
    # axarr[1].show()
    #
    # f, _ = pu.plot_results(results, average_group=True, split_fn=lambda _: '', shaded_std=False)
    # f.show()
    # pu.plot_results(results, average_group=True)
