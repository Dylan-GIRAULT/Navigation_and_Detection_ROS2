from setuptools import find_packages, setup
from glob import glob

package_name = 'shared'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", glob("launch/*.launch.xml")),
        (f"share/{package_name}/resource", glob("resource/*")),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='dylan',
    maintainer_email='dylan.girault.etu.utc.fr',
    description='TODO: Package description',
    license='MIT License',
    extras_require={
        'test': [
            'pytest',
        ],
    },
)
