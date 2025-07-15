import napari
from napari_orientationpy._widget import OrientationWidget
from napari_orientationpy import __version__

if __name__ == "__main__":
    viewer = napari.Viewer(title=f"Napari OrientationPy ({__version__})")
    viewer.window.add_dock_widget(OrientationWidget(viewer), name="OrientationPy")
    napari.run()