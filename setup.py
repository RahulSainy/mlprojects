# Importing necessary functions from the setuptools library
from setuptools import find_packages, setup
# Importing List type from the typing module to specify return type
from typing import List

# !important to include "-e ." in requirements.txt to trigger setup.py correctly
HYPEN_E_DOT = "-e ."

def get_requirements(file_path: str) -> List[str]:
    '''
    Reads the requirements from a specified file and returns them as a list.
    '''
    requirements = []
    
    # Opening the specified file_path (like 'requirements.txt')
    with open(file_path) as file_obj:
        # Reading all the lines from requirements.txt into a list
        requirements = file_obj.readlines()
        
        # Stripping newline characters from each requirement
        requirements = [req.strip() for req in requirements]
        
        # Removing "-e ." if it's there to prevent setup.py from running into itself
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
            
    return requirements

# Configuration settings for setup.py
setup(
    # Name of the project
    name='mlproject',
    # Version of the project
    version='0.1',
    # Author of the project
    author='Rahul Saini',
    
    # Finding all packages to include (looking for directories with __init__.py files)
    packages=find_packages(),
    
    # Specifying the minimum required Python version
    python_requires='>=3.7',
    
    # List of dependencies to install, obtained from requirements.txt
    install_requires=get_requirements('requirements.txt'),
    # Using the get_requirements function to fetch requirements from requirements.txt
)
