import os
from glob import glob

from setuptools import setup

package_name = "primitive_camera_tools"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", [os.path.join("resource", package_name)]),
        (os.path.join("share", package_name), ["package.xml"]),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
        (os.path.join("share", package_name, "sounds"), glob("sounds/*")),
        (os.path.join("share", package_name, "launch"), glob("launch/*.xml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="nuri",
    maintainer_email="nuri.ulusoy@robeff.com",
    description="Camera frame publisher and emergency sound player.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "camera_frame_publisher = primitive_camera_tools.camera_frame_publisher:main",
            "emergency_sound_player = primitive_camera_tools.emergency_sound_player:main",
        ],
    },
)
