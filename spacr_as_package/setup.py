from setuptools import setup, find_packages
import subprocess, sys, re

def get_cuda_version():
    """
    Attempts to find the CUDA version installed on the system by invoking the nvcc command.
    Returns:
        A string representing the CUDA version (e.g., '11.2'), or None if CUDA is not found.
    """
    try:
        # Query the CUDA compiler version
        nvcc_output = subprocess.check_output(["nvcc", "--version"], text=True)
        
        # Use regular expression to extract version number
        version_match = re.search(r"release (\d+\.\d+)", nvcc_output)
        if version_match:
            return version_match.group(1)
    except subprocess.CalledProcessError as e:
        print(f"Failed to execute nvcc command: {e}")
    except FileNotFoundError:
        print("nvcc command not found. Ensure CUDA is installed and nvcc is in your PATH.")
    
    # Return None if CUDA is not found or an error occurs
    return None

def install_torch(cuda_version):
    """
    Installs the appropriate PyTorch version based on the detected CUDA version or installs the CPU version if CUDA is not detected.
    """
    # Map CUDA version to PyTorch wheel tags or use 'cpu' for CPU-only installation
    cuda_to_wheel_tag = {
        "11.8": "cu118",
        "12.1": "cu121",
        "cpu": "cpu"
    }

    # Determine the appropriate wheel tag based on the detected CUDA version
    wheel_tag = cuda_to_wheel_tag.get(cuda_version, "cpu")

    # Specify the desired PyTorch version
    pytorch_version = "2.2.0"
    torchvision_version = "0.17.0"
    torchaudio_version = "2.2.0"

    # Construct the base installation command
    base_command = f"pip3 install torch=={pytorch_version} torchvision=={torchvision_version} torchaudio=={torchaudio_version}"

    # Append the appropriate index-url based on the wheel tag, if necessary
    if wheel_tag in ["cu118", "cu121"]:
        command = f"{base_command} --index-url https://download.pytorch.org/whl/{wheel_tag}"
    elif wheel_tag == "cpu":
        command = f"{base_command} --index-url https://download.pytorch.org/whl/cpu"
    else:
        # Default to the general installation command without specifying CUDA version
        command = base_command

    print(f"Running command: {command}")
    try:
        subprocess.run(command, check=True, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("PyTorch installation completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to install PyTorch: {e.output.decode()}")


#PYTORCH_VERSION = "2.2.0"
#TORCHVISION_VERSION = "0.17.0"
#TORCHAUDIO_VERSION = "2.2.0"

cuda_version = get_cuda_version()
if cuda_version:
    print(f"CUDA version detected: {cuda_version}")
else:
    print("CUDA not detected.")

dependencies = [
    'numpy>=1.21.0',
    'pandas>=1.3.0',
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
    'opencv-python',
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