import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'pingpong'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Enzo',
    maintainer_email='esmajlaj@olin.edu',
    description='Ping Pong Player',
    license='All rights reserved.',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'ball_detector = pingpong.ball_detector:main',
            'trajectory_estimator = pingpong.trajectory_estimator:main',
            'intercept_planner = pingpong.intercept_planner:main',
        ],
    },
)
