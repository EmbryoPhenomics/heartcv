from setuptools import setup
import codecs
import os.path
import pathlib

here = pathlib.Path(__file__).parent.resolve()
long_description = (here / "README.md").read_text(encoding="utf-8")

# For reading in the version string without importing the package
# Ref: https://packaging.python.org/guides/single-sourcing-package-version/
def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), "r") as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


setup(
    name="heartcv",
    version=get_version("heartcv/__init__.py"),
    description="A platform for automated, noninvasive extraction of various cardiac proxy traits from bioimaging footage.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/EmbryoPhenomics/heartcv",
    author="Ziad Ibbini, Oliver Tills",
    author_email="ziad.ibbini@students.plymouth.ac.uk, oli.tills@plymouth.ac.uk",
    license="MIT",
    packages=["heartcv"],
    python_requires=">=3.5, <4",
    install_requires=[
        "vuba",
        "numpy",
        "scipy",
        "tqdm",
        "opencv-python",
        "more_itertools"
    ],
    project_urls={
        "Source": "https://github.com/EmbryoPhenomics/heartcv/tree/main/heartcv"
    },
)
