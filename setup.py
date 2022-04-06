from setuptools import find_packages, setup

install_requires = [line.rstrip() for line in open("requirements.txt", "r")]

setup(name='tacto_learn',
      version='0.1.0',
      description='Learning algorithms for tacto environments',
      author='Tianyu Wang',
      author_email="tiw161@eng.ucsd.edu",
      packages=find_packages(),
      install_requires=install_requires,
      python_requires=">=3.8",
)