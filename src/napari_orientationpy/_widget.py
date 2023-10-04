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
    QFileDialog,
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
        self.sigma = 2.0
        self.mode = 'fiber'
        self.orientation_computed = False

        # Layout
        grid_layout = QGridLayout()
        grid_layout.setAlignment(Qt.AlignTop)
        self.setLayout(grid_layout)

        # Image
        self.cb_image = QComboBox()
        self.cb_image.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        grid_layout.addWidget(QLabel("Image (2D, 3D or RGB)", self), 0, 0)
        grid_layout.addWidget(self.cb_image, 0, 1)

        # Sigma
        self.sigma_spinbox = QDoubleSpinBox()
        self.sigma_spinbox.setMinimum(0.1)
        self.sigma_spinbox.setValue(self.sigma)
        self.sigma_spinbox.setSingleStep(0.1)
        self.sigma_spinbox.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        grid_layout.addWidget(QLabel("Structural scale (sigma)", self), 1, 0)
        grid_layout.addWidget(self.sigma_spinbox, 1, 1)

        # Mode
        self.cb_mode = QComboBox()
        self.cb_mode.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.cb_mode.addItems([self.mode, 'membrane'])
        grid_layout.addWidget(QLabel("Mode", self), 2, 0)
        grid_layout.addWidget(self.cb_mode, 2, 1)

        grid_layout.addWidget(QLabel("Output color-coded orientation", self), 3, 0)
        self.cb_rgb = QCheckBox()
        self.cb_rgb.setChecked(False)
        grid_layout.addWidget(self.cb_rgb, 3, 1)

        grid_layout.addWidget(QLabel("Output orientation gradient", self), 4, 0)
        self.cb_origrad = QCheckBox()
        self.cb_origrad.setChecked(False)
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
        self.cb_vec.stateChanged.connect(self._enable_vector_params)
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

        # # Vector scale (outdated)
        # self.vector_scale_spinbox = QDoubleSpinBox()
        # self.vector_scale_spinbox.setMinimum(0.0)
        # self.vector_scale_spinbox.setMaximum(100.0)
        # self.vector_scale_spinbox.setValue(1.0)
        # self.vector_scale_spinbox.setSingleStep(0.05)
        # self.vector_scale_spinbox.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        # vectors_layout.addWidget(QLabel("Length factor", self), 4, 0)
        # vectors_layout.addWidget(self.vector_scale_spinbox, 4, 1)

        # Energy rescaling
        vectors_layout.addWidget(QLabel("Length is energy", self), 5, 0)
        self.cb_energy_rescale = QCheckBox()
        self.cb_energy_rescale.setChecked(False)
        vectors_layout.addWidget(self.cb_energy_rescale, 5, 1)

        # Compute button
        self.compute_orientation_btn = QPushButton("Compute orientation", self)
        self.compute_orientation_btn.clicked.connect(self._trigger_compute_orientation)
        grid_layout.addWidget(self.compute_orientation_btn, 10, 0, 1, 2)

        # Save button
        self.save_orientation_btn = QPushButton("Save orientation (CSV)", self)
        self.save_orientation_btn.clicked.connect(self._save_orientation)
        self.save_orientation_btn.setEnabled(self.orientation_computed)
        grid_layout.addWidget(self.save_orientation_btn, 11, 0, 1, 2)

        # Progress bar
        self.pbar = QProgressBar(self, minimum=0, maximum=1)
        self.pbar.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        grid_layout.addWidget(self.pbar, 12, 0, 1, 2)

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
                    # if not x.rgb:
                    self.cb_image.addItem(x.name, x.data)
    
    def _enable_vector_params(self, value):
        self.node_spacing_spinbox_X.setEnabled(value != 0)
        self.node_spacing_spinbox_Y.setEnabled(value != 0)
        self.node_spacing_spinbox_Z.setEnabled(value != 0)
        self.cb_energy_rescale.setEnabled(value != 0)

    @property
    def ndims(self):
        if self.image is not None:
            return len(self.image.shape)

    def _save_orientation(self):
        import pandas as pd
        node_origins = np.stack([g for g in np.mgrid[[slice(0, x) for x in self.image.shape]]])
        node_origins = node_origins.reshape(self.ndims, -1).T
        dim_headers = ['X', 'Y', 'Z'][:self.ndims]
        df = pd.DataFrame(data=node_origins, columns=dim_headers)
        if self.theta is not None: df['theta'] = self.theta.ravel()
        if self.phi is not None: df['phi'] = self.phi.ravel()
        if self.energy is not None: df['energy'] = self.energy.ravel()
        if self.coherency is not None: df['coherency'] = self.coherency.ravel()
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(self, "Save CSV File", "", "CSV Files (*.csv);;All Files (*)", options=options)
        if not file_path.endswith('.csv'): file_path += '.csv'
        df.to_csv(file_path, index=False)

    def _orientation_vectors(self):
        """
        Computes and displays orientation vectors on a regular spatial grid (2D or 3D).
        """
        node_spacings = [
            self.node_spacing_spinbox_X.value(),
            self.node_spacing_spinbox_Y.value(),
            self.node_spacing_spinbox_Z.value(),
        ][:self.ndims][::-1]  # ZYX order in 3D
        # slices = [slice(0, None, n) for n in node_spacings]
        slices = [slice(n // 2, None, n) for n in node_spacings]
        node_origins = np.stack([g[tuple(slices)] for g in np.mgrid[[slice(0, x) for x in self.image.shape]]])
        energy_sample = self.energy[tuple(slices)]
        slices.insert(0, slice(0, None))
        displacements = self.boxVectorCoords[tuple(slices)].copy()
        displacements *= np.mean(node_spacings)
        # displacements *= self.vector_scale_spinbox.value()  # Outdated
        if self.cb_energy_rescale.isChecked():
            displacements *= energy_sample
        displacements = np.reshape(displacements, (self.ndims, -1)).T
        origins = np.reshape(node_origins, (self.ndims, -1)).T
        origins = origins - displacements / 2
        displacement_vectors = np.stack((origins, displacements))
        displacement_vectors = np.rollaxis(displacement_vectors, 1)

        vector_props = {
            'name': 'Orientation vectors',
            'edge_width': np.max(node_spacings) / 5.0,
            'opacity': 1.0,
            'ndim': self.ndims,
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
        image_layer = self.viewer.layers[self.cb_image.currentText()]
        if image_layer is None:
            return
        
        # If RGB, convert the image to grayscale
        self.image = np.mean(image_layer.data, axis=2) if image_layer.rgb else image_layer.data
        
        if (self.ndims == 2) & (self.cb_mode.currentText() != 'fiber'):
            self.cb_mode.setCurrentIndex(0)
            show_info('Set mode to fiber (2D image).')
        self.mode = self.cb_mode.currentText()
        self.sigma = self.sigma_spinbox.value()

        gradients = orientationpy.computeGradient(self.image, mode='splines')
        structureTensor = orientationpy.computeStructureTensor(gradients, sigma=self.sigma)
        orientation_returns = orientationpy.computeOrientation(
            structureTensor, 
            mode=self.mode,
            computeEnergy=True, 
            computeCoherency=True,
        )
        if not self.orientation_computed:
            self.orientation_computed = True
            self.save_orientation_btn.setEnabled(self.orientation_computed)
        self.boxVectorCoords = orientationpy.anglesToVectors(orientation_returns)
        self.theta = orientation_returns.get('theta') + 90
        self.phi = orientation_returns.get('phi')
        self.energy = rescale_intensity(orientation_returns.get('energy'), out_range=(0, 1))
        self.coherency = rescale_intensity(orientation_returns.get('coherency'), out_range=(0, 1))

        if self.ndims == 3:
            imDisplayHSV = np.stack((self.phi / 360, np.sin(np.deg2rad(self.theta)), self.image / self.image.max()), axis=-1)
        elif self.ndims == 2:
            imDisplayHSV = np.stack((self.theta / 180, self.coherency, self.image / self.image.max()), axis=-1)
        else:
            print(f'Number of dimensions ({self.ndims}) not supported.')
        self.imdisplay_rgb = matplotlib.colors.hsv_to_rgb(imDisplayHSV)

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