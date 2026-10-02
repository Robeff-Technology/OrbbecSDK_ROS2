import os
from glob import glob

from setuptools import setup

package_name = "dock_target_detector"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", [os.path.join("resource", package_name)]),
        (os.path.join("share", package_name), ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.xml")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml") + glob("config/*.rviz")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="nuri",
    maintainer_email="nuri.ulusoy@robeff.com",
    description="AprilTag dock target detection and pose estimation for reverse docking.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "dock_detector = dock_target_detector.dock_detector:main",
            "make_dock_target = dock_target_detector.make_dock_target:main",
            "dock_monitor = dock_target_detector.dock_monitor:main",
            "camera_reset = dock_target_detector.camera_reset:main",
        ],
    },
)
