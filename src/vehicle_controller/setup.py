from setuptools import find_packages, setup

package_name = 'vehicle_controller'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config/', ['config/params.yaml'])
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='gyeongrak',
    maintainer_email='gyeongrak@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'mc_test_01 = vehicle_controller.mc_test_01_takeoff_landing:main',
            'mc_test_02 = vehicle_controller.mc_test_02_bezier_triangle:main',
            'mc_test_03 = vehicle_controller.mc_test_03_offboard_mission:main',
            'mc_test_04 = vehicle_controller.mc_test_04_track_hover:main',
            'mc_main = vehicle_controller.mc_main:main',
            'mc_main_temp = vehicle_controller.mc_main_temp:main',
            'precise_landing = vehicle_controller.precise_landing_test:main',
            'precise_landing_basket = vehicle_controller.precise_landing_basket_test:main',
            'test_logger = vehicle_controller.mc_test_logger:main',
        ],
    },
)
