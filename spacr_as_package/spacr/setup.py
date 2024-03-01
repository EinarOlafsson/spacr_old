from setuptools import setup, find_packages

dependencies = [
    'torch', 
    'numpy',
    'pandas',
    'statsmodels',
    'scikit-image',
    'scikit-learn',
    'seaborn',
    'matplotlib',
    'pillow',
    'imageio',
    'scipy',
    'ipywidgets',
    'mahotas',
    'btrack',
    'trackpy',
    'cellpose',
    'IPython',
    'opencv-python-headless',
]

setup(
    name="spacr",
    version="0.0.1",
    author="Your Name",
    author_email="olafsson@med.umich.com",
    description="A brief description of your package",
    long_description=open('README.md').read(),
    url="https://github.com/EinarOlafsson/spacr",
    packages=find_packages(exclude=["tests.*", "tests"]),
    install_requires=dependencies,
    extras_require={
        'dev': ['pytest>=3.9'],
        'headless': ['opencv-python-headless'],
        'full': ['opencv-python'],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)