from typing import TYPE_CHECKING

from napari_tools_menu import register_dock_widget
from qtpy.QtWidgets import QVBoxLayout, QWidget, QComboBox, QSizePolicy, QLabel, QGridLayout, QSlider, QPushButton
from qtpy.QtCore import Qt

if TYPE_CHECKING:
    import napari


import orientationpy
import napari
import matplotlib
import numpy as np
import napari.layers
from skimage.exposure import rescale_intensity


@register_dock_widget(menu="Orientationpy > Orientation")
class OrientationWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer

        import skimage.data
        sample_3d_image = np.repeat(skimage.data.coins()[:70, :70][None], 30, axis=0)
        self.viewer.add_image(sample_3d_image, name='image 1')
        self.viewer.add_image(-1 * sample_3d_image, name='image 2')

        self.vector_scale = 10.0
        self.orientation_computed = False
        self.image = None
        self.phi = self.theta = self.energy = self.coherency = None

        # Layout
        grid_layout = QGridLayout()
        self.setLayout(grid_layout)

        # Image
        self.cb_image = QComboBox()
        self.cb_image.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        grid_layout.addWidget(QLabel("Image", self), 0, 0)
        grid_layout.addWidget(self.cb_image, 0, 1)
        self.select_btn = QPushButton("Select", self)
        self.select_btn.clicked.connect(self._layer_selected)
        grid_layout.addWidget(self.select_btn, 0, 2)

        # State of the selection
        self.image_name = ''
        self.selection = QLabel(f"Selected: {self.image_name}", self)
        grid_layout.addWidget(self.selection, 1, 0, 1, 2)

        # Compute orientation
        self.compute_orientation_btn = QPushButton("Compute orientation", self)
        self.compute_orientation_btn.clicked.connect(self._compute_orientation)
        grid_layout.addWidget(self.compute_orientation_btn, 2, 0, 1, 2)

        # Show RGB orientation
        self.show_image_rgb_btn = QPushButton("Show RGB", self)
        self.show_image_rgb_btn.clicked.connect(self._compute_imDisplayRGB)
        grid_layout.addWidget(self.show_image_rgb_btn, 3, 0, 1, 2)

        # Show vectors
        self.show_vectors_btn = QPushButton("Show vectors", self)
        self.show_vectors_btn.clicked.connect(self._compute_vectors)
        grid_layout.addWidget(self.show_vectors_btn, 4, 0, 1, 2)

        # Setup layer callbacks
        self.viewer.layers.events.inserted.connect(
            lambda e: e.value.events.name.connect(self._on_layer_change)
        )
        self.viewer.layers.events.inserted.connect(self._on_layer_change)
        self.viewer.layers.events.removed.connect(self._on_layer_change)
        self._on_layer_change(None)

    def _layer_selected(self):
        self.image_name = self.cb_image.currentText()
        self.selection.setText(f"Selected: {self.image_name}")

        self.image = self.cb_image.currentData()
        self.orientation_computed = False
        self.phi = self.theta = self.energy = self.coherency = None

    def _on_layer_change(self, e):
        self.cb_image.clear()
        for x in self.viewer.layers:
            if isinstance(x, napari.layers.Image):
                self.cb_image.addItem(x.name, x.data)

    def _set_vector_scale(self, vector_scale: float=10.0):
        self.vector_scale = vector_scale

    def _compute_vectors(self):
        if not self.orientation_computed:
            print('Please compute the orientation first.')
            return
        

        node_spacing=(1, 10, 10)

        
        phi_sample = self.phi[::(node_spacing[0]), ::(node_spacing[1]), ::(node_spacing[2])]
        theta_sample = self.theta[::(node_spacing[0]), ::(node_spacing[1]), ::(node_spacing[2])]

        rx, ry, rz = self.image.shape

        xgrid, ygrid, zgrid = np.mgrid[0:rx, 0:ry, 0:rz]
        x_sample = xgrid[::(node_spacing[0]), ::(node_spacing[1]), ::(node_spacing[2])]
        y_sample = ygrid[::(node_spacing[0]), ::(node_spacing[1]), ::(node_spacing[2])]
        z_sample = zgrid[::(node_spacing[0]), ::(node_spacing[1]), ::(node_spacing[2])]

        node_origins = np.concatenate((x_sample[None], y_sample[None], z_sample[None]))

        theta_radians = np.radians(theta_sample)
        phi_radians = np.radians(phi_sample)

        x = np.cos(theta_radians) * np.sin(phi_radians)
        y = np.sin(theta_radians) * np.sin(phi_radians)
        z = np.cos(phi_radians)
        (displacements_cartesian := np.concatenate((x[None], y[None], z[None]))) / np.linalg.norm(displacements_cartesian, axis=0)

        displacements_cartesian *= self.vector_scale

        energy_rescaled = self.energy[::(node_spacing[0]), ::(node_spacing[1]), ::(node_spacing[2])]
        displacements_cartesian *= (energy_normalized := rescale_intensity(energy_rescaled, out_range=(0, 1)))

        a = np.reshape(node_origins, (3, -1)).T[None]
        b = np.reshape(displacements_cartesian, (3, -1)).T[None]
        displacement_vectors = np.vstack((a, b))
        displacement_vectors = np.rollaxis(displacement_vectors, 1)

        ### Add vectors ###
        vector_props = {
            'name': f'vectors_ns_{node_spacing}',
            'edge_width': 0.7,
            'opacity': 1.0,
            'ndim': 3,
            'features': {'length': energy_normalized.ravel()},
            'edge_color': 'length',
            'vector_style': 'line',
        }

        self.viewer.add_vectors(displacement_vectors, **vector_props)

        # return displacement_vectors, energy_normalized
    
    def _compute_orientation(self) -> np.ndarray:

        if self.image is None:
            print("Select an image first.")
            return

        sigma = 1.0

        gx, gy, gz = orientationpy.computeGradient(self.image, mode='splines')
        structureTensor = orientationpy.computeStructureTensor((gx, gy, gz), sigma=sigma)
        orientation_returns = orientationpy.computeOrientation(
            structureTensor, 
            mode="membrane",
            computeEnergy=True, 
            computeCoherency=True,
        )

        self.theta = orientation_returns['theta']
        self.phi = orientation_returns['phi']
        self.energy = orientation_returns['energy']
        self.coherency = orientation_returns['coherency']

        self.orientation_computed = True

        print(f'Orientation computed!')

    def _compute_imDisplayRGB(self):
        if not self.orientation_computed:
            print('Please compute the orientation first.')
            return
        
        rx, ry, rz = self.image.shape
        imDisplayHSV = np.zeros((rx, ry, rz, 3), dtype="f4")
        imDisplayHSV[..., 0] = self.phi / 360
        imDisplayHSV[..., 1] = np.sin(np.deg2rad(self.theta))
        imDisplayHSV[..., 2] = self.image / self.image.max()

        imDisplayRGB = matplotlib.colors.hsv_to_rgb(imDisplayHSV)

        ### Add RGB image ###

        self.viewer.add_image(
            imDisplayRGB,
            rgb=True,
            name="Color-coded orientation"
        )

        # return imDisplayRGB