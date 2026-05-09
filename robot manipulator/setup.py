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
        		"lidar_avoidance_v2 = first_package.lidar_avoidance_v2:main",
        		"pid_controller = first_package.pid_controller:main",
        		"self_balance_lqr = first_package.self_balance_lqr:main",
        		"arm_pid = first_package.arm_pid:main",
        		'ct_pid = first_package.ct_pid:main',
        		'pid_gravity_node = first_package.PİD_Gravity:main',
        		'robust_gravity = first_package.robust_gravity:main',
        		'adaptive_controller = first_package.adaptive_controller:main',
        		'FLNN_Controller = first_package.FLNN_Controller:main',

        ],
    },
)
