from typing import TYPE_CHECKING

from napari_tools_menu import register_dock_widget
from qtpy.QtWidgets import (
    QWidget, 
    QComboBox, 
    QSizePolicy, 
    QLabel, 
    QGridLayout, 
    QPushButton,
    QSpinBox,
    QDoubleSpinBox,
    QCheckBox,
    QProgressBar,
)
from qtpy.QtCore import Qt

if TYPE_CHECKING:
    import napari

import orientationpy
import napari
from napari.qt.threading import thread_worker
import matplotlib
import numpy as np
import napari.layers
from skimage.exposure import rescale_intensity


@register_dock_widget(menu="Orientationpy > Orientation (pixels)")
class OrientationWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer

        self.image = None
        self.phi = self.theta = self.energy = self.coherency = None
        self.imdisplay_rgb = None
        self.sigma = 10
        self.mode = 'fiber'

        # Layout
        grid_layout = QGridLayout()
        grid_layout.setAlignment(Qt.AlignTop)
        self.setLayout(grid_layout)

        # Image
        self.cb_image = QComboBox()
        self.cb_image.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        grid_layout.addWidget(QLabel("3D Image", self), 0, 0)
        grid_layout.addWidget(self.cb_image, 0, 1)

        # Sigma
        self.sigma_spinbox = QDoubleSpinBox()
        self.sigma_spinbox.setMinimum(0.1)
        self.sigma_spinbox.setValue(self.sigma)
        self.sigma_spinbox.setSingleStep(0.1)
        self.sigma_spinbox.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        grid_layout.addWidget(QLabel("Sigma", self), 1, 0)
        grid_layout.addWidget(self.sigma_spinbox, 1, 1)

        # Mode
        self.cb_mode = QComboBox()
        self.cb_mode.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.cb_mode.addItems([self.mode, 'membrane'])
        grid_layout.addWidget(QLabel("Mode", self), 2, 0)
        grid_layout.addWidget(self.cb_mode, 2, 1)

        grid_layout.addWidget(QLabel("Output RGB", self), 3, 0)
        self.cb_rgb = QCheckBox()
        self.cb_rgb.setChecked(True)
        grid_layout.addWidget(self.cb_rgb, 3, 1)

        grid_layout.addWidget(QLabel("Output vectors", self), 4, 0)
        self.cb_vec = QCheckBox()
        self.cb_vec.setChecked(True)
        grid_layout.addWidget(self.cb_vec, 4, 1)

        # Vector node spacing
        self.node_spacing_spinbox = QSpinBox()
        self.node_spacing_spinbox.setMinimum(1)
        self.node_spacing_spinbox.setMaximum(100)
        self.node_spacing_spinbox.setValue(10)
        self.node_spacing_spinbox.setSingleStep(1)
        self.node_spacing_spinbox.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        grid_layout.addWidget(QLabel("Node spacing (XY)", self), 5, 0)
        grid_layout.addWidget(self.node_spacing_spinbox, 5, 1)

        self.compute_orientation_btn = QPushButton("Compute orientation", self)
        self.compute_orientation_btn.clicked.connect(self._trigger_compute_orientation)
        grid_layout.addWidget(self.compute_orientation_btn, 6, 0, 1, 2)

        # Progress bar
        self.pbar = QProgressBar(self, minimum=0, maximum=1)
        self.pbar.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        grid_layout.addWidget(self.pbar, 7, 0, 1, 2)

        # Setup layer callbacks
        self.viewer.layers.events.inserted.connect(
            lambda e: e.value.events.name.connect(self._on_layer_change)
        )
        self.viewer.layers.events.inserted.connect(self._on_layer_change)
        self.viewer.layers.events.removed.connect(self._on_layer_change)
        self._on_layer_change(None)

    def _on_layer_change(self, e):
        self.cb_image.clear()
        for x in self.viewer.layers:
            if isinstance(x, napari.layers.Image):
                if len(x.data.shape) == 3:
                    self.cb_image.addItem(x.name, x.data)

    def _orientation_vectors(self):
        
        ns = self.node_spacing_spinbox.value()
        vector_scale = ns * 2

        node_spacing=(1, ns, ns)

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

        displacements_cartesian *= vector_scale

        energy_rescaled = self.energy[::(node_spacing[0]), ::(node_spacing[1]), ::(node_spacing[2])]
        displacements_cartesian *= (energy_normalized := rescale_intensity(energy_rescaled, out_range=(0, 1)))

        a = np.reshape(node_origins, (3, -1)).T[None]
        b = np.reshape(displacements_cartesian, (3, -1)).T[None]
        displacement_vectors = np.vstack((a, b))
        displacement_vectors = np.rollaxis(displacement_vectors, 1)

        vector_props = {
            # 'name': f'vectors_ns_{node_spacing}',
            'name': 'Orientation vectors',
            'edge_width': 0.7,
            'opacity': 1.0,
            'ndim': 3,
            'features': {'length': energy_normalized.ravel()},
            'edge_color': 'length',
            'vector_style': 'line',
        }

        for layer in self.viewer.layers:
            if layer.name == "Orientation vectors":
                layer.data = displacement_vectors
                return

        self.viewer.add_vectors(displacement_vectors, **vector_props)

    @thread_worker
    def _fake_worker(self):
        import time; time.sleep(0.5)

    @thread_worker
    def _compute_orientation(self) -> np.ndarray:
        self.image = self.cb_image.currentData()
        self.sigma = self.sigma_spinbox.value()
        self.mode = self.cb_mode.currentText()

        gx, gy, gz = orientationpy.computeGradient(self.image, mode='splines')
        structureTensor = orientationpy.computeStructureTensor((gx, gy, gz), sigma=self.sigma)
        orientation_returns = orientationpy.computeOrientation(
            structureTensor, 
            mode=self.mode,
            computeEnergy=True, 
            computeCoherency=False,
        )

        self.theta = orientation_returns['theta']
        self.phi = orientation_returns['phi']
        self.energy = orientation_returns['energy']
        # self.coherency = orientation_returns['coherency']

        # Done once, by default.
        rx, ry, rz = self.image.shape
        imDisplayHSV = np.zeros((rx, ry, rz, 3), dtype="f4")
        imDisplayHSV[..., 0] = self.phi / 360
        imDisplayHSV[..., 1] = np.sin(np.deg2rad(self.theta))
        imDisplayHSV[..., 2] = self.image / self.image.max()

        self.imdisplay_rgb = matplotlib.colors.hsv_to_rgb(imDisplayHSV)

    def _imdisplay_rgb(self):        
        for layer in self.viewer.layers:
            if layer.name == "Color-coded orientation":
                layer.data = self.imdisplay_rgb
                return
        
        self.viewer.add_image(self.imdisplay_rgb, rgb=True, name="Color-coded orientation")

    def _trigger_compute_orientation(self):
        self.pbar.setMaximum(0)
        if self._check_should_compute():
            worker = self._compute_orientation()
        else:
            worker = self._fake_worker()
        worker.returned.connect(self._thread_returned)
        worker.start()

    def _check_should_compute(self):
        if self.cb_image.currentData() is None:
            print("Select an image first.")
            return False
        
        if self.sigma != self.sigma_spinbox.value():
            return True
        
        if self.mode != self.cb_mode.currentText():
            return True

        if self.image is None:
            return True
        
        if not np.array_equal(self.cb_image.currentData(), self.image):
            return True
        
        return False

    def _thread_returned(self):
        if self.cb_rgb.isChecked(): self._imdisplay_rgb()
        if self.cb_vec.isChecked(): self._orientation_vectors()
        self.pbar.setMaximum(1)