from setuptools import setup, find_packages

setup(
    name='eigencwd',
    version='1.0',
    description='EigenCWD: a spatially varying deconvolution algorithm for single metalens imaging',
    packages=['eigencwd'],
    package_dir={'':'src'},
    author="Joel Yeo",
    author_email="joelyeo@u.nus.edu",
    license="GPL v3.0",
    keywords="deconvolution, deblurring, metalens",
)