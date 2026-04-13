from setuptools import find_packages, setup

package_name = 'first_package'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    package_data={'': ['py.typed']},
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='erhan',
    maintainer_email='erhan@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': ['move_turtle = first_package.move_turtle:main',
        		"get_position = first_package.get_position:main",
        		"move_and_stop = first_package.move_and_stop:main",
        		"lidar_avoidance = first_package.lidar_avoidance:main",
        ],
    },
)
