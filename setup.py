from typing import List
from setuptools import find_packages, setup

HYPHEN_E_DOT = '-e .'

def get_requirements(file_path: str) -> List[str]:
    """
    This function will return the list of requirements from requirements.txt
    """
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        # Remove newlines
        requirements = [req.replace("\n", "") for req in requirements]

        # Remove '-e .' if present, as it triggers setup.py itself
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
            
    return requirements

setup(
    name='recommendation_system',
    version='0.1.0',
    author='Ahmad',
    author_email='realahmadmalik3@gmail.com',
    # Keep 'src' if your folders are inside a src/ directory
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=get_requirements('requirements.txt'),
    entry_points={
        'console_scripts': [
            'train-recsys=pipeline.train_pipeline:main',
            'predict-recsys=pipeline.predict_pipeline:main',
        ],
    },
)