from setuptools import setup

license = ""
version = ""

with open("../../LICENSE", encoding="utf-8") as f:
    license = "\n" + f.read()

line = open("../version.txt", encoding="utf-8").readlines()[0]
version = line.rstrip('\n')

setup(
    name = "mvnc",
    version = version,
    author = "Intel Corporation",
    description = ("mvnc python api"),
    license = license,
    keywords = "",
    url = "http://developer.movidius.com",
    packages = ["mvnc"],
    package_dir = {"mvnc": "mvnc"},
    install_requires = [
        "numpy",
    ],
    long_description = "-",
    classifiers = [
        "Development Status :: 5 - Production/Stable",
        "Topic :: Software Development :: Libraries",
        "License :: Other/Proprietary License",
    ],
)
