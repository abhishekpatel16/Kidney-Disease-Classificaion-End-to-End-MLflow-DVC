import setuptools  # Importing the setuptools module for package setup

# Reading the content of README.md to use it as the long description
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

# Defining the package version
__version__ = "0.0.0"

# Defining metadata for the package
REPO_NAME = "Kidney-Disease-Classificaion-End-to-End-MLflow-DVC"  # GitHub repository name
AUTHOR_USER_NAME = "abhishekpatel16"  # GitHub username of the author
SRC_REPO = "cnnClassifier"  # Name of the source package
AUTHOR_EMAIL = "abhishekpatel0771@gmail.com"  # Author's email

# Setting up the package using setuptools
setuptools.setup(
    name=SRC_REPO,  # Package name (same as the source directory)
    version=__version__,  # Version of the package
    author=AUTHOR_USER_NAME,  # Author's name
    author_email=AUTHOR_EMAIL,  # Author's email address
    description="A small python package for CNN app",  # Short description of the package
    long_description=long_description,  # Detailed description from README.md
    long_description_content="text/markdown",  # Format of the long description
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",  # URL of the GitHub repository
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",  # URL for reporting issues
    },
    package_dir={"": "src"},  # Defining the root directory for packages
    packages=setuptools.find_packages(where="src")  # Automatically find packages in the "src" directory
)
