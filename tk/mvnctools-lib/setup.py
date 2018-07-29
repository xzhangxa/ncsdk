import re
from setuptools import setup, find_packages

license = ""
version = ""

with open("../../LICENSE", encoding="utf-8") as f:
    license = "\n" + f.read()

for line in open("../version.txt", encoding="utf-8"):
    m = re.search("mvnctools-lib[^\w]*?([\d\.]+)", line)
    if m:
        version = m.group(1)
        break

setup(
    name = "mvnctools-lib",
    version = version,
    author = "Intel Corporation",
    description = ("mvnc python toolkits - library"),
    license = license,
    keywords = "",
    url = "http://developer.movidius.com",
    packages = [
        "mvnctools.Controllers",
        "mvnctools.Controllers.Parsers",
        "mvnctools.Controllers.Parsers.Parser",
        "mvnctools.Controllers.Parsers.CaffeParser",
        "mvnctools.Controllers.Parsers.TensorFlowParser",
        "mvnctools.Models",
        "mvnctools.Models.StageDefinitions",
        "mvnctools.Views",
    ],
    install_requires = [
        "numpy",
        "graphviz",
        "pygraphviz",
        "protobuf",
        "PyYAML",
        "Pillow",
        "scipy",
        "scikit-image",
        "ete3",
        "tensorflow==1.4.0",
        "mvnc",
    ],
    long_description = "-",
    classifiers = [
        "Development Status :: 5 - Production/Stable",
        "Topic :: Software Development :: Build Tools",
        "License :: Other/Proprietary License",
    ],
    zip_safe = False,
)
