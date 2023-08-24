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
    QProgressBar,
)
from qtpy.QtCore import Qt

if TYPE_CHECKING:
    import napari

import orientationpy
import napari
from napari.qt.threading import thread_worker
import numpy as np
import napari.layers
from skimage.exposure import rescale_intensity

@register_dock_widget(menu="Orientationpy > Orientation (boxes)")
class OrientationBoxesWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer

        self.image = None
        self.mode = 'fiber'
        self.nsx = self.nsy = 10
        self.nsz = 1

        # Layout
        grid_layout = QGridLayout()
        grid_layout.setAlignment(Qt.AlignTop)
        self.setLayout(grid_layout)

        # Image
        self.cb_image = QComboBox()
        self.cb_image.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        grid_layout.addWidget(QLabel("3D Image", self), 0, 0)
        grid_layout.addWidget(self.cb_image, 0, 1)

        # Mode
        self.cb_mode = QComboBox()
        self.cb_mode.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.cb_mode.addItems([self.mode, 'membrane'])
        grid_layout.addWidget(QLabel("Mode", self), 1, 0)
        grid_layout.addWidget(self.cb_mode, 1, 1)

        # Node spacing X
        self.node_spacing_spinbox_X = QSpinBox()
        self.node_spacing_spinbox_X.setMinimum(1)
        self.node_spacing_spinbox_X.setMaximum(100)
        self.node_spacing_spinbox_X.setValue(self.nsx)
        self.node_spacing_spinbox_X.setSingleStep(1)
        self.node_spacing_spinbox_X.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        grid_layout.addWidget(QLabel("Node spacing (X)", self), 2, 0)
        grid_layout.addWidget(self.node_spacing_spinbox_X, 2, 1)

        # Node spacing Y
        self.node_spacing_spinbox_Y = QSpinBox()
        self.node_spacing_spinbox_Y.setMinimum(1)
        self.node_spacing_spinbox_Y.setMaximum(100)
        self.node_spacing_spinbox_Y.setValue(self.nsy)
        self.node_spacing_spinbox_Y.setSingleStep(1)
        self.node_spacing_spinbox_Y.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        grid_layout.addWidget(QLabel("Node spacing (Y)", self), 3, 0)
        grid_layout.addWidget(self.node_spacing_spinbox_Y, 3, 1)

        # Node spacing Z
        self.node_spacing_spinbox_Z = QSpinBox()
        self.node_spacing_spinbox_Z.setMinimum(1)
        self.node_spacing_spinbox_Z.setMaximum(100)
        self.node_spacing_spinbox_Z.setValue(self.nsz)
        self.node_spacing_spinbox_Z.setSingleStep(1)
        self.node_spacing_spinbox_Z.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        grid_layout.addWidget(QLabel("Node spacing (Z)", self), 4, 0)
        grid_layout.addWidget(self.node_spacing_spinbox_Z, 4, 1)

        # Compute button
        self.compute_orientation_btn = QPushButton("Compute orientation", self)
        self.compute_orientation_btn.clicked.connect(self._trigger_compute_orientation)
        grid_layout.addWidget(self.compute_orientation_btn, 5, 0, 1, 2)

        # Progress bar
        self.pbar = QProgressBar(self, minimum=0, maximum=1)
        self.pbar.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        grid_layout.addWidget(self.pbar, 6, 0, 1, 2)

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

    @thread_worker
    def _fake_worker(self):
        import time; time.sleep(0.5)

    @thread_worker
    def _compute_orientation(self) -> np.ndarray:
        self.image = self.cb_image.currentData()
        self.mode = self.cb_mode.currentText()

        self.nsx = self.node_spacing_spinbox_X.value()
        self.nsy = self.node_spacing_spinbox_Y.value()
        self.nsz = self.node_spacing_spinbox_Z.value()

        node_spacing=(self.nsz, self.nsy, self.nsx)

        structureTensorBoxes = orientationpy.computeGradientStructureTensorBoxes(
            self.image,
            node_spacing,
        )

        orientation_returns = orientationpy.computeOrientation(
            structureTensorBoxes,
            mode=self.mode,
            computeEnergy=True,
            computeCoherency=False,
        )

        thetaBoxes = orientation_returns['theta']
        energyBoxes = orientation_returns['energy']

        boxVectorsZYX = orientationpy.anglesToVectors(orientation_returns)

        boxCentresX, boxCentresY, boxCentresZ = np.mgrid[
            self.nsz // 2 : thetaBoxes.shape[0] * self.nsz + self.nsz // 2 : self.nsz,
            self.nsy // 2 : thetaBoxes.shape[1] * self.nsy + self.nsy // 2 : self.nsy,
            self.nsx // 2 : thetaBoxes.shape[2] * self.nsx + self.nsx // 2 : self.nsx,
        ]

        boxCentres = np.concatenate((boxCentresX[None], boxCentresY[None], boxCentresZ[None]), axis=0)

        _, bx, by, bz = boxCentres.shape

        bc = boxCentres.reshape((3, bx*by*bz))
        bv = boxVectorsZYX.reshape((3, bx*by*bz))

        energy_normalized = rescale_intensity(energyBoxes, out_range=(0, 1))
        bv *= energy_normalized.reshape((bx*by*bz))
        bv *= np.mean([self.nsz, self.nsy, self.nsx])  # mean?

        vectors = np.concatenate((bc[None], bv[None]), axis=0)
        vectors = np.rollaxis(vectors, axis=2)

        return (vectors, energy_normalized)

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
        
        if self.nsx != self.node_spacing_spinbox_X.value():
            return True
        
        if self.nsy != self.node_spacing_spinbox_Y.value():
            return True
        
        if self.nsz != self.node_spacing_spinbox_Z.value():
            return True
        
        if self.mode != self.cb_mode.currentText():
            return True

        if self.image is None:
            return True
        
        return False
        
    def _thread_returned(self, payload):
        self.pbar.setMaximum(1)

        displacement_vectors, energy_normalized = payload

        vector_props = {
            'name': 'Orientation boxes',
            'edge_width': 0.7,
            'opacity': 1.0,
            'ndim': 3,
            'features': {'length': energy_normalized.ravel()},
            'edge_color': 'length',
            'vector_style': 'line',
        }

        for layer in self.viewer.layers:
            if layer.name == "Orientation boxes":
                layer.data = displacement_vectors
                return

        self.viewer.add_vectors(displacement_vectors, **vector_props)
