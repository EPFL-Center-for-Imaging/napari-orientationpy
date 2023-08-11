from napari_orientationpy import OrientationWidget
import skimage.data
import numpy as np

def test_widget(make_napari_viewer):
    viewer = make_napari_viewer()

    sample_3d_image = np.repeat(skimage.data.coins()[:70, :70][None], 30, axis=0)
    viewer.add_image(sample_3d_image)

    widget = OrientationWidget(viewer)

    run_btn = widget.compute_orientation_btn
    run_btn.click()

    assert len(viewer.layers) == 3, f'Number of layers is : {len(viewer.layers)}'

