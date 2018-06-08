import re
from setuptools import setup

license = ""
version = ""

with open("../../LICENSE", encoding="utf-8") as f:
    license = "\n" + f.read()

for line in open("../../VERSION", encoding="utf-8"):
    m = re.search("mvnc[^\w]*?([\d\.]+)", line)
    if m:
        version = m.group(1)
        break

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
