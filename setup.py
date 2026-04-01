from setuptools import setup, find_packages
from glob import glob
import os
import logging

logging.getLogger("setuptools_scm").setLevel(logging.CRITICAL)
logging.getLogger("setuptools_scm._file_finders").setLevel(logging.CRITICAL)
logging.getLogger("setuptools_scm._file_finders.git").setLevel(logging.CRITICAL)

package_name = 'capstone_pkg'

def data_files_from_dir(root_dir: str):
    data_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        if not filenames:
            continue
        rel = os.path.relpath(dirpath, start=root_dir)   # '.' or 'subdir/...'
        install_dir = os.path.join('share', package_name, root_dir, rel if rel != '.' else '')
        files = [os.path.join(dirpath, f) for f in filenames]
        data_files.append((install_dir, files))
    return data_files
setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'numpy', 'opencv-python'],
    zip_safe=True,
    maintainer='gaga',
    maintainer_email='fhmpsy@gmail.com',
    description='TODO: Package description',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'main = capstone_pkg.main:main',
            'simulation = capstone_pkg.simulation:main',
            'realsense = capstone_pkg.camera.view_realsense:main',
            'zed = capstone_pkg.camera.view_zed:main',
            'curobo_ik = capstone_pkg.kinematics.curobo_test_ik:main',
            'bidir_rrt = capstone_pkg.main:main_birrt',
            'impedance = capstone_pkg.impedance.impedance:main',
        ],
    },
)
