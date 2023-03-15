import orientationpy as opy
import numpy as np
import matplotlib.colors
import matplotlib.pyplot as plt

try:
    import napari
except:
    raise ImportError('It looks like you do not have Napari installed.')

import tifffile
import os
import sys

def read_image(file: str) -> np.ndarray:
    """Basic image reader."""
    ext = os.path.splitext(file)[-1]
    print(ext)
    if ext in ['.tif', '.tiff']:
        return tifffile.imread(file)
    elif ext == '.npy':
        return np.load(file)
    else:
        raise Exception('Could not read this image.')

def main(file):
    image = read_image(file)
    image = image.astype(float) / 256

    boxSizePixels = 3

    structureTensorBoxes = opy.computeGradientStructureTensorBoxes(image, [boxSizePixels]*3)
    dicoBoxes = opy.computeOrientation(structureTensorBoxes, computeEnergy=True)

    thetaBoxes = dicoBoxes['theta']
    # phiBoxes = dicoBoxes['phi']
    energyBoxes = dicoBoxes['energy']

    boxVectorsZYX = opy.anglesToVectors(dicoBoxes)

    boxCentresX, boxCentresY, boxCentresZ = np.mgrid[
        boxSizePixels // 2 : thetaBoxes.shape[0] * boxSizePixels + boxSizePixels // 2 : boxSizePixels,
        boxSizePixels // 2 : thetaBoxes.shape[1] * boxSizePixels + boxSizePixels // 2 : boxSizePixels,
        boxSizePixels // 2 : thetaBoxes.shape[2] * boxSizePixels + boxSizePixels // 2 : boxSizePixels,
    ]

    boxCentres = np.concatenate((boxCentresX[None], boxCentresY[None], boxCentresZ[None]), axis=0)

    _, bx, by, bz = boxCentres.shape

    bc = boxCentres.reshape((3, bx*by*bz))
    bv = boxVectorsZYX.reshape((3, bx*by*bz))

    # Rescale according to energy
    bv *= energyBoxes.reshape((bx*by*bz)) / energyBoxes.max()
    bv *= boxSizePixels

    vectors = np.concatenate((bc[None], bv[None]), axis=0)
    vectors = np.rollaxis(vectors, axis=2)

    viewer = napari.Viewer(ndisplay=3)
    viewer.add_image(image)
    print(thetaBoxes.shape)
    viewer.add_vectors(  # (N, 2, 3)
        vectors, edge_width=0.4, edge_color='orientation', features={'orientation': thetaBoxes.ravel()},
    )

    viewer.camera.angles = (0, 15, 35)
    sc = viewer.screenshot(canvas_only=True, flash=False)

    # fig, ax = plt.subplots(figsize=(7, 7))
    # ax.imshow(sc)
    # ax.axis('off')
    # plt.tight_layout()
    # plt.show()

if __name__ == '__main__':
    _, image_file = sys.argv
    main(image_file)
    napari.run()