import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ied-localize",
    version="0.1",
    author="C. Price Withers",
    author_email="price.withers@nih.gov",
    description="Source Localization of IEDs using GM and WM",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/witherscp/ied-localize.git",
    packages=setuptools.find_packages(include=["ied_localize", "ied_localize.*"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: Linux/Unix",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy",
        "pandas",
        "nilearn",
        "matplotlib",
        "seaborn",
        "pygeodesic",
        "jaro-winkler",
        "scipy",
    ],
    scripts=["ied_localize/do_localize_source.py"],
)
