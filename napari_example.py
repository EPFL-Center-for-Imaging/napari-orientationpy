import orientationpy as opy
import numpy as np
import matplotlib.colors
import matplotlib.pyplot as plt

try:
    import napari
    from napari.experimental import link_layers
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

    viewer = napari.Viewer(ndisplay=3)

    gradients = opy.computeGradient(image)
    structureTensor = opy.computeStructureTensor(gradients, sigma=2.0)
    orientations = opy.computeOrientation(structureTensor)
    
    theta = orientations.get('theta')
    phi = orientations.get('phi')

    rx, ry, rz = image.shape
    imDisplayHSV = np.zeros((rx, ry, rz, 3), dtype="f4")
    imDisplayHSV[..., 0] = phi / 360
    imDisplayHSV[..., 1] = np.sin(np.deg2rad(theta))
    imDisplayHSV[..., 2] = image / image.max()

    imDisplayRGB = matplotlib.colors.hsv_to_rgb(imDisplayHSV)

    red, green, blue = np.rollaxis(imDisplayRGB, axis=-1)
    red_channel_layer = viewer.add_image(red, blending='additive', colormap='red')
    green_channel_layer = viewer.add_image(green, blending='additive', colormap='green')
    blue_channel_layer = viewer.add_image(blue, blending='additive', colormap='blue')
    link_layers([red_channel_layer, green_channel_layer, blue_channel_layer])

    viewer.camera.angles = (0, 15, 35)
    sc = viewer.screenshot(canvas_only=True, flash=False)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(sc)
    ax.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    _, image_file = sys.argv
    main(image_file)
    napari.run()