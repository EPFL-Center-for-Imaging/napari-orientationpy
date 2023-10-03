from typing import TYPE_CHECKING

from napari.utils.notifications import show_info
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
    QGroupBox,
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

from .misorientation import fast_misorientation_angle

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
        grid_layout.addWidget(QLabel("Image (2D or 3D)", self), 0, 0)
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
        self.cb_rgb.setChecked(False)
        grid_layout.addWidget(self.cb_rgb, 3, 1)

        grid_layout.addWidget(QLabel("Output gradient", self), 4, 0)
        self.cb_origrad = QCheckBox()
        self.cb_origrad.setChecked(True)
        grid_layout.addWidget(self.cb_origrad, 4, 1)

        ### Vectors group
        vectors_group = QGroupBox(self)
        vectors_layout = QGridLayout()
        vectors_group.setLayout(vectors_layout)
        vectors_group.layout().setContentsMargins(10, 10, 10, 10)
        grid_layout.addWidget(vectors_group, 5, 0, 1, 2)

        # Output vectors
        vectors_layout.addWidget(QLabel("Output vectors", self), 0, 0)
        self.cb_vec = QCheckBox()
        self.cb_vec.setChecked(True)
        vectors_layout.addWidget(self.cb_vec, 0, 1)

        # Vector display spacing (X)
        self.node_spacing_spinbox_X = QSpinBox()
        self.node_spacing_spinbox_X.setMinimum(1)
        self.node_spacing_spinbox_X.setMaximum(100)
        self.node_spacing_spinbox_X.setValue(1)
        self.node_spacing_spinbox_X.setSingleStep(1)
        self.node_spacing_spinbox_X.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        vectors_layout.addWidget(QLabel("Spacing (X)", self), 1, 0)
        vectors_layout.addWidget(self.node_spacing_spinbox_X, 1, 1)

        # Vector display spacing (Y)
        self.node_spacing_spinbox_Y = QSpinBox()
        self.node_spacing_spinbox_Y.setMinimum(1)
        self.node_spacing_spinbox_Y.setMaximum(100)
        self.node_spacing_spinbox_Y.setValue(1)
        self.node_spacing_spinbox_Y.setSingleStep(1)
        self.node_spacing_spinbox_Y.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        vectors_layout.addWidget(QLabel("Spacing (Y)", self), 2, 0)
        vectors_layout.addWidget(self.node_spacing_spinbox_Y, 2, 1)

        # Vector display spacing (Z)
        self.node_spacing_spinbox_Z = QSpinBox()
        self.node_spacing_spinbox_Z.setMinimum(1)
        self.node_spacing_spinbox_Z.setMaximum(100)
        self.node_spacing_spinbox_Z.setValue(1)
        self.node_spacing_spinbox_Z.setSingleStep(1)
        self.node_spacing_spinbox_Z.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        vectors_layout.addWidget(QLabel("Spacing (Z)", self), 3, 0)
        vectors_layout.addWidget(self.node_spacing_spinbox_Z, 3, 1)

        # Vector scale
        self.vector_scale_spinbox = QDoubleSpinBox()
        self.vector_scale_spinbox.setMinimum(0.0)
        self.vector_scale_spinbox.setMaximum(100.0)
        self.vector_scale_spinbox.setValue(1.0)
        self.vector_scale_spinbox.setSingleStep(0.05)
        self.vector_scale_spinbox.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        vectors_layout.addWidget(QLabel("Length factor", self), 4, 0)
        vectors_layout.addWidget(self.vector_scale_spinbox, 4, 1)

        # Energy rescaling
        vectors_layout.addWidget(QLabel("Length is energy", self), 5, 0)
        self.cb_energy_rescale = QCheckBox()
        self.cb_energy_rescale.setChecked(True)
        vectors_layout.addWidget(self.cb_energy_rescale, 5, 1)

        # Compute button
        self.compute_orientation_btn = QPushButton("Compute orientation", self)
        self.compute_orientation_btn.clicked.connect(self._trigger_compute_orientation)
        grid_layout.addWidget(self.compute_orientation_btn, 10, 0, 1, 2)

        # Progress bar
        self.pbar = QProgressBar(self, minimum=0, maximum=1)
        self.pbar.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        grid_layout.addWidget(self.pbar, 11, 0, 1, 2)

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
                if len(x.data.shape) in [2, 3]:
                    if not x.rgb:
                        self.cb_image.addItem(x.name, x.data)

    def _orientation_vectors(self):
        """
        Computes and displays orientation vectors on a regular spatial grid.
        """
        self.nsx = self.node_spacing_spinbox_X.value()
        self.nsy = self.node_spacing_spinbox_Y.value()
        self.nsz = self.node_spacing_spinbox_Z.value()

        ndims = len(self.image.shape)
        is_3D = ndims == 3

        energy_normalized = rescale_intensity(self.energy, out_range=(0, 1))

        if is_3D:
            vector_scale = np.mean([self.nsz, self.nsy, self.nsx])
            xgrid, ygrid, zgrid = np.mgrid[0:self.image.shape[0], 0:self.image.shape[1], 0:self.image.shape[2]]
            node_origins = np.stack(
                (
                    xgrid[::self.nsz, ::self.nsy, ::self.nsx], 
                    ygrid[::self.nsz, ::self.nsy, ::self.nsx], 
                    zgrid[::self.nsz, ::self.nsy, ::self.nsx]
                )
            )
            energy_sample = energy_normalized[::self.nsz, ::self.nsy, ::self.nsx]
            boxVectorCoords = orientationpy.anglesToVectors(self.orientation_returns)
            displacements_cartesian = boxVectorCoords[:, ::self.nsz, ::self.nsy, ::self.nsx]
        else:
            vector_scale = np.mean([self.nsy, self.nsx])
            xgrid, ygrid = np.mgrid[0:self.image.shape[0], 0:self.image.shape[1]]
            node_origins = np.stack(
                (
                    xgrid[::self.nsy, ::self.nsx], 
                    ygrid[::self.nsy, ::self.nsx]
                )
            )
            energy_sample = energy_normalized[::self.nsy, ::self.nsx]
            boxVectorCoords = orientationpy.anglesToVectors(self.orientation_returns)
            displacements_cartesian = boxVectorCoords[:, ::self.nsy, ::self.nsx]
        
        displacements_cartesian *= vector_scale
        displacements_cartesian *= self.vector_scale_spinbox.value()
        if self.cb_energy_rescale.isChecked():
            displacements_cartesian *= energy_sample

        displacements = np.reshape(displacements_cartesian, (ndims, -1)).T[None]
        origins = np.reshape(node_origins, (ndims, -1)).T[None] - displacements / 2

        displacement_vectors = np.vstack((origins, displacements))
        displacement_vectors = np.rollaxis(displacement_vectors, 1)

        edge_width = np.max([self.nsx, self.nsy, self.nsz]) / 5.0
        vector_props = {
            'name': 'Orientation vectors',
            'edge_width': edge_width,
            'opacity': 1.0,
            'ndim': ndims,
            'features': {'length': energy_sample.ravel()},
            'edge_color': 'length',
            'vector_style': 'line',
        }

        for idx, layer in enumerate(self.viewer.layers):
            if layer.name == "Orientation vectors":
                self.viewer.layers.pop(idx)

        self.viewer.add_vectors(displacement_vectors, **vector_props)

    @thread_worker
    def _fake_worker(self):
        import time; time.sleep(0.5)

    @thread_worker
    def _compute_orientation(self) -> np.ndarray:
        """
        Computes the greylevel orientations of the image.
        """
        self.image = self.cb_image.currentData()
        image_shape = self.image.shape
        is_3D = len(image_shape) == 3
        if not is_3D:
            if self.cb_mode.currentText() != 'fiber':
                self.cb_mode.setCurrentIndex(0)
                show_info('Set mode to fiber (2D image).')
        self.mode = self.cb_mode.currentText()
        self.sigma = self.sigma_spinbox.value()

        gradients = orientationpy.computeGradient(self.image, mode='splines')
        structureTensor = orientationpy.computeStructureTensor(gradients, sigma=self.sigma)
        self.orientation_returns = orientationpy.computeOrientation(
            structureTensor, 
            mode=self.mode,
            computeEnergy=True, 
            computeCoherency=True,
        )

        self.theta = self.orientation_returns.get('theta') + 90
        self.phi = self.orientation_returns.get('phi')
        self.energy = self.orientation_returns.get('energy')
        self.coherency = self.orientation_returns.get('coherency')

        if is_3D:
            rx, ry, rz = image_shape
            imDisplayHSV = np.zeros((rx, ry, rz, 3), dtype="f4")
            imDisplayHSV[..., 0] = self.phi / 360
            imDisplayHSV[..., 1] = np.sin(np.deg2rad(self.theta))
            imDisplayHSV[..., 2] = self.image / self.image.max()
        else:
            rx, ry = image_shape
            imDisplayHSV = np.zeros((rx, ry, 3), dtype="f4")
            imDisplayHSV[..., 0] = (self.theta) / 180
            imDisplayHSV[..., 1] = self.coherency / self.coherency.max()
            imDisplayHSV[..., 2] = self.image / self.image.max()
        
        self.imdisplay_rgb = matplotlib.colors.hsv_to_rgb(imDisplayHSV)

        # Orientation gradient
        self.orientation_gradient = fast_misorientation_angle(self.theta, self.phi)

    def _orientation_gradient(self):
        for layer in self.viewer.layers:
            if layer.name == "Orientation gradient":
                layer.data = self.orientation_gradient
                return

        self.viewer.add_image(self.orientation_gradient, colormap="inferno", name="Orientation gradient", blending="additive")

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
            show_info("Select an image first.")
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
        if self.cb_image.currentData() is not None:
            if self.cb_rgb.isChecked(): self._imdisplay_rgb()
            if self.cb_vec.isChecked(): self._orientation_vectors()
            if self.cb_origrad.isChecked(): self._orientation_gradient()
        else:
            show_info("Select an image first.")
        self.pbar.setMaximum(1)