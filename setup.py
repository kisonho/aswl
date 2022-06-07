from distutils.core import setup

setup(name='aswl',
      version='0.1.2',
      description='Learning Pruned Structure and Weights Simultaneously from Scratch: an Attention based Approach',
      author='Kison Ho',
      author_email='unfit-gothic.0q@icloud.com',
      packages=[
            'aswl'
      ],
      package_dir={
            'aswl': 'aswl'
      },
      install_requires=[
            'tensorflow'
      ],
      python_requires=">=3.8",
      url="https://github.com/kisonho/torchmanager.git"
)
