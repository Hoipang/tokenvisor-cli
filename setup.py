from setuptools import setup

setup(
    name="mipod-cli",
    version="0.1.0",
    packages=["mipod_cli"],
    install_requires=[
        # List your dependencies here
        "pyyaml>=6.0.2",
        "requests>=2.32.0",
        "click>=8.0.0",  # Example if using Click
    ],
    entry_points={
        "console_scripts": [
            "mipod-cli=mipod_cli.main:cli",
        ],
    },
    author="Embedded LLM",
    description="AMD GPU Pod scheduling CLI to deploy & serve LLM.",
    license="MIT",
)
