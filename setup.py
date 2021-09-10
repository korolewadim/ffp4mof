import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ffp4mof",
    version="1.0.0",
    author="Vadim Korolev",
    author_email="korolewadim@gmail.com",
    description="A small example package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/korolewadim/ffp4mof",
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,
    packages=setuptools.find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "ase==3.22.0",
        "scikit-learn==0.24.1",
        "pymatgen==2021.3.9",
        "matminer==0.6.5",
        "xgboost==1.1.1",
    ],
)