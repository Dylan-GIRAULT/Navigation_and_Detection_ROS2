from setuptools import find_packages, setup
from glob import glob

package_name = 'screen'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (f"share/{package_name}/launch", glob("launch/*.launch.xml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer='jules',
    maintainer_email='jules.gatelier.etu.utc.fr',
    description="Package used for display purposes",
    license="Proprietary",
    tests_require=["pytest"],
    entry_points={
        'console_scripts': [
            "control_display_node = screen.screen_control:main",
            "map_display_node = screen.screen_map:main",
            "perc_int_node = screen.screen_perc_int:main",
        ],
    },
)
