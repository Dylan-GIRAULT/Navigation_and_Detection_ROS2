from setuptools import find_packages, setup
from glob import glob

package_name = 'perc_int'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (f"share/{package_name}/launch", glob("launch/*.launch.xml")),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jules',
    maintainer_email='jules.gatelier.etu.utc.fr',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        "console_scripts": [
            "perception_interieure_gesture = perc_int.launch_gesture:main",
            "perception_interieure_focus = perc_int.launch_focus:main",
            "perception_interieure = perc_int.launch:main",
        ],
    },
)
