from __future__ import annotations

from pathlib import Path

from setuptools import find_packages, setup

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # Python <3.11
    import tomli as tomllib


def read_pyproject():
    with open("pyproject.toml", "rb") as f:
        data = tomllib.load(f)
    return data["project"]


project = read_pyproject()

setup(
    name=project["name"],
    version=project["version"],
    description=project.get("description", ""),
    long_description=Path("README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    author=project.get("authors", [{}])[0].get("name", ""),
    license=project.get("license", {}).get("text", ""),
    keywords=project.get("keywords", []),
    classifiers=project.get("classifiers", []),
    python_requires=project.get("requires-python"),
    install_requires=project.get("dependencies", []),
    extras_require=project.get("optional-dependencies", {}),
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    project_urls=project.get("urls", {}),
)
