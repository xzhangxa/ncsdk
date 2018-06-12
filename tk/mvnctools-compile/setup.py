import re
from setuptools import setup

license = ""
version = ""

with open("../../LICENSE", encoding="utf-8") as f:
    license = "\n" + f.read()

for line in open("../../VERSION", encoding="utf-8"):
    m = re.search("mvnctools-compile[^\w]*?([\d\.]+)", line)
    if m:
        version = m.group(1)
        break

setup(
    name = "mvnctools-compile",
    version = version,
    author = "Intel Corporation",
    description = ("mvnc python toolkits - compile"),
    license = license,
    keywords = "",
    url = "http://developer.movidius.com",
    scripts = [
        "mvnctools/mvNCCompile",
    ],
    install_requires = [
        "numpy",
        "mvnctools-lib",
    ],
    long_description = "-",
    classifiers = [
        "Development Status :: 5 - Production/Stable",
        "Topic :: Software Development :: Build Tools",
        "License :: Other/Proprietary License",
    ],
    zip_safe = False,
)
