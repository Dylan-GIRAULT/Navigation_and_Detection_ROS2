import os
from glob import glob
from setuptools import setup

package_name = 'localisation'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (f"share/{package_name}/launch", glob("launch/*.launch.xml")),
        ('share/' + package_name, ['resource/better_circuit_enu_simplify.csv']),
        ],
    
    install_requires=['setuptools', 'numpy', 'pyproj','scipy'],
    zip_safe=True,
    maintainer='hugo',
    maintainer_email='hugo.huyet-dumong@etu.utc.fr',
    description='Matching Node',
    license='Proprietary',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'icp_node = localisation.main:main',
        ],
    },
)
