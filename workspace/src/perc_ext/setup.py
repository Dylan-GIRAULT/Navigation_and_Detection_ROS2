import os
from glob import glob
from setuptools import setup

package_name = 'perc_ext'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        
        # --- CORRECTION CHEMINS (D'après votre ls -R) ---
        # 1. Le dossier model
        (os.path.join('share', package_name, 'model'), glob('perc_ext/model/*.pth')),
        
        # 2. Le dossier cfgs (et ses sous-dossiers importants)
        # Note: glob ne fait pas de récursif par défaut, on ajoute le fichier principal
        (os.path.join('share', package_name, 'cfgs'), glob('perc_ext/cfgs/*.yaml')),
        
        # Si vous avez besoin des sous-dossiers (kitti_models, etc), il faut les ajouter
        # Mais pour l'instant, le fichier pointpillar.yaml est à la racine de cfgs, donc ça suffit.
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='younes',
    maintainer_email='younes@todo.todo',
    description='Road Segmentation Node',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # 1. L'ANCIEN (Ne touchez pas, ça pointe vers node_percext.py)
            'detecter_route = perc_ext.node_percext:main',
            
            # 2. LE NOUVEAU (Pointe vers le fichier que vous venez de créer)
            'road_segmentation_node = perc_ext.road_segmentation_node:main',
        ],
    },
)
