from pathlib import Path

from setuptools import find_packages, setup

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="information_systems",
    version="0.0.1",
    description="Analysis on multiple graph datasets with Graph embeddings and GNNs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Spiros Maggioros, Rei Pasai, Eleni Nasopoyloy",
    author_email="spirosmag@ieee.org",
    maintainer="Spiros Maggioros, Rei Pasai, Eleni Nasopoyloy",
    maintainer_email="spirosmag@ieee.org",
    download_url="https://github.com/spirosmaggioros/information_systems/",
    url="https://github.com/spirosmaggioros/information_systems/",
    packages=find_packages(exclude=[".github"]),
    python_requires=">=3.9",
    install_requires=[
        "networkx",
        "torchvision",
        "torch-geometric",
        "karateclub",
        "tqdm",
        "numpy",
        "pandas",
        "scipy<1.13.0",
        "scikit-learn",
    ],
    entry_points={
        "console_scripts": ["information_systems = information_systems.__main__:main"]
    },
    license="By installing/using information_systems, the user agrees to the MIT license",
    keywords=[
        "Graph2Vec",
        "NetLSD",
        "GIN",
        "Graph Embeddings",
    ],
    package_data={"information_systems": ["VERSION"]},
)
