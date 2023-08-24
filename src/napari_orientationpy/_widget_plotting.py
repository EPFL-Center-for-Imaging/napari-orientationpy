from typing import TYPE_CHECKING

from napari_tools_menu import register_dock_widget
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

from napari_orientationpy._plotting import plotOrientations3d

@register_dock_widget(menu="Orientationpy > Orientation (plot)")
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

        self.cb_plot = QComboBox()
        self.cb_plot.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.cb_plot.addItems(["points", "bins"])
        grid_layout.addWidget(QLabel("Plot type", self), 1, 0)
        grid_layout.addWidget(self.cb_plot, 1, 1)

        self.cb_projection = QComboBox()
        self.cb_projection.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.cb_projection.addItems(["lambert", "equidistant", "stereo"])
        grid_layout.addWidget(QLabel("Projection", self), 2, 0)
        grid_layout.addWidget(self.cb_projection, 2, 1)

        self.plot_btn = QPushButton("Plot orientation distribution")
        self.plot_btn.clicked.connect(self._plot_orientation)
        grid_layout.addWidget(self.plot_btn, 3, 0, 1, 2)

        self.canvas = FigureCanvas()
        self.canvas.figure.set_tight_layout(True)
        self.canvas.figure.patch.set_facecolor("#262930")
        self.axes = self.canvas.figure.subplots()
        self.canvas.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        self.canvas.setMinimumSize(250, 250)
        grid_layout.addWidget(self.canvas, 4, 0, 1, 2)

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
                self.cb_vectors.addItem(x.name, x.data)

    def _plot_orientation(self):
        vectors_data = self.cb_vectors.currentData()
        if vectors_data is None:
            return
        
        vector_displacements = vectors_data[:, 1]

        # print(vector_displacements[0])
        # vector_displacements = vector_displacements[:, ::-1]
        # print(vector_displacements[0])

        # z = vector_displacements[:, 0]
        # y = vector_displacements[:, 1]
        # x = vector_displacements[:, 2]

        # import matplotlib.pyplot as plt
        # plt.scatter(x, y)
        # plt.show()

        projection = self.cb_projection.currentText()
        plot = self.cb_plot.currentText()

        self.axes.cla()

        plotOrientations3d(
            orientations_zyx=vector_displacements,
            projection=projection,
            plot=plot,
            ax=self.axes,
        )

        self.canvas.draw()