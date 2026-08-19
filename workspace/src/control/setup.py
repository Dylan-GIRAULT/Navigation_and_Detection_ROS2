from setuptools import find_packages, setup
from glob import glob

package_name = "control"

setup(
    name=package_name,
    version="2.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", glob("launch/*.launch.xml")),
        (f"share/{package_name}/data", glob("data/*.csv")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer='jules',
    maintainer_email='jules.gatelier.etu.utc.fr',
    description="Control Package that send messages to move the vehicle",
    license="Proprietary",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "control_node = control.control:main",
            "simulation_node = control.simulation:main"
        ],
    },
)