from setuptools import find_packages, setup

package_name = 'vision_processing_nodes'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='bulnabi',
    maintainer_email='thomas426789@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'image_processing_node = vision_processing_nodes.image_processing_node:main',
            'image_processing_node_gazebo = vision_processing_nodes.image_processing_node_gazebo:main',
            'camera_publisher_node = vision_processing_nodes.camera_publisher_node:main',

        ],
    },
)
