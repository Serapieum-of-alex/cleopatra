from setuptools import find_packages, setup

with open("README.md", "r") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()

requirements = [line.strip() for line in open("requirements.txt").readlines()]
requirements_dev = [line.strip() for line in open("requirements-dev.txt").readlines()]

setup(
    name="cleopatra",
    version="0.4.1",
    description="visualization package",
    author="Mostafa Farrag",
    author_email="moah.farag@gmail.come",
    url="https://github.com/MAfarrag/cleopatra",
    keywords=["matplotlib", "visualization"],
    long_description=readme + "\n\n" + history,
    repository="https://github.com/MAfarrag/celopatra",
    documentation="https://celopatra.readthedocs.io/",
    long_description_content_type="text/markdown",
    license="GNU General Public License v3",
    zip_safe=False,
    packages=find_packages(include=["cleopatra", "cleopatra.*"]),
    test_suite="tests",
    tests_require=requirements_dev,
    install_requires=requirements,
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Environment :: Console",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Visualization",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
    ],
)
