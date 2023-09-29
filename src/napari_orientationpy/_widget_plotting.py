from typing import TYPE_CHECKING

from qtpy.QtWidgets import (
    QWidget, 
    QComboBox, 
    QSizePolicy, 
    QLabel, 
    QGridLayout, 
    QPushButton,
)
from qtpy.QtCore import Qt

if TYPE_CHECKING:
    import napari

import napari
import napari.layers
from matplotlib.backends.backend_qt5agg import FigureCanvas

from napari_orientationpy._plotting import stereo_plot

class OrientationPlottingWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer

        grid_layout = QGridLayout()
        grid_layout.setAlignment(Qt.AlignTop)
        self.setLayout(grid_layout)

        self.cb_vectors = QComboBox()
        self.cb_vectors.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        grid_layout.addWidget(QLabel("3D Vectors", self), 0, 0)
        grid_layout.addWidget(self.cb_vectors, 0, 1)

        self.cb_direction = QComboBox()
        self.cb_direction.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.cb_direction.addItems(["Z", "Y", "X"])
        grid_layout.addWidget(QLabel("Direction", self), 1, 0)
        grid_layout.addWidget(self.cb_direction, 1, 1)

        self.plot_btn = QPushButton("Plot orientation distribution")
        self.plot_btn.clicked.connect(self._plot_orientation)
        grid_layout.addWidget(self.plot_btn, 2, 0, 1, 2)

        self.canvas = FigureCanvas()

        self.canvas.figure.set_tight_layout(True)
        self.canvas.figure.set_size_inches(6.0, 6.0)
        self.canvas.figure.patch.set_facecolor("#5a626c")

        self.axes = self.canvas.figure.subplots()
        self.axes.set_facecolor("#ffffff")
        self.axes.set_xticks([])
        self.axes.set_yticks([])
        self.axes.set_xlabel('X')
        self.axes.set_ylabel('Y')
        self.axes.xaxis.label.set_color('white')
        self.axes.yaxis.label.set_color('white')

        self.canvas.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        # self.canvas.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        self.canvas.setMinimumSize(200, 200)
        grid_layout.addWidget(self.canvas, 3, 0, 1, 2)

        self.viewer.layers.events.inserted.connect(
            lambda e: e.value.events.name.connect(self._on_layer_change)
        )
        self.viewer.layers.events.inserted.connect(self._on_layer_change)
        self.viewer.layers.events.removed.connect(self._on_layer_change)
        self._on_layer_change(None)

    def _on_layer_change(self, e):
        self.cb_vectors.clear()
        for x in self.viewer.layers:
            if isinstance(x, napari.layers.Vectors):
                if x.data.shape[2] == 3:
                    self.cb_vectors.addItem(x.name, x.data)

    def _plot_orientation(self):
        vectors_data = self.cb_vectors.currentData()
        if vectors_data is None:
            return
        
        self.axes.cla()
        self.canvas.figure.patch.set_facecolor("#5a626c")
        self.axes.set_facecolor("#ffffff")
        self.axes.set_xticks([])
        self.axes.set_yticks([])
        self.axes.xaxis.label.set_color('white')
        self.axes.yaxis.label.set_color('white')

        stereo_plot(
            self.axes,
            vector_displacements=vectors_data[:, 1],
            direction=self.cb_direction.currentText(),
            sample_size=min(len(vectors_data), 4000),
        )

        self.canvas.draw()