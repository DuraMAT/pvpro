import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pvpro",
    version="0.0.2",
    author="toddkarin",
    author_email="pvtools.lbl@gmail.com",
    description="Extract single diode model paramaters from PV power plant production data.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/DuraMAT/pvpro",
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=['numpy','pandas','scipy'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)