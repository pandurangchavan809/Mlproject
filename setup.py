from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT = '-e .'
/
def get_requirements(file_path: str) -> List[str]:
    """
    This function returns the list of requirements
    """
    requirements: List[str] = []
    with open(file_path) as file_obj:
        for line in file_obj:
            # Remove inline comments and surrounding whitespace
            req = line.split('#', 1)[0].strip()
            if not req:
                continue
            # Skip editable installs (e.g. -e .)
            if req.startswith('-e'):
                continue
            requirements.append(req)

    return requirements


setup(
    name='mlproject',
    version='0.0.1',
    author='Pc',
    author_email='pandurangchavan809@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)