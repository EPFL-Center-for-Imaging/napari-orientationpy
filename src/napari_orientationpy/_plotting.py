import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def stereo_plot(ax, vector_displacements, direction='Z', sample_size=4000):
    """
    Stereo projection of vector directions.
    """
    sample = vector_displacements[np.random.choice(np.arange(sample_size), size=sample_size)]
    norms = np.apply_along_axis(np.linalg.norm, 1, sample)
    vects_normed = sample / norms.reshape(-1, 1)
    
    projection_xy = np.zeros((len(vects_normed), 2))
    idx = {'Z': 0, 'Y': 1, 'X': 2}[direction]
    vects_normed[vects_normed[:, idx] > 0] *= -1  # Take care of symmetry
    for k, vect in enumerate(np.roll(vects_normed, axis=1, shift=-idx)):
        if vect[0] == 1:
            continue
        projection_xy[k] = [
            vect[1] / (1 - vect[0]), 
            vect[2] / (1 - vect[0])
        ]

    # ax.set_facecolor((1.0, 1.0, 1.0))
    ax.add_artist( plt.Circle((0, 0), 1, fill=False))
    ax.set_xlim(-1.01, 1.01)
    ax.set_ylim(-1.01, 1.01)

    # print(idx)
    # print(titles)
    titles = np.roll(np.array(['Z' ,'Y', 'X']), shift=-idx)
    ax.set_ylabel(titles[1])
    ax.set_xlabel(titles[2])

    # ax.set_axis_off()
    sns.kdeplot(
        pd.DataFrame({'X': projection_xy[:, 0], 'Y': projection_xy[:, 1]}), 
        x='X', y='Y', fill=True, ax=ax
    )
    # ax.plot(vects_normed[:, 0], vects_normed[:, 1], ".", markersize=5)