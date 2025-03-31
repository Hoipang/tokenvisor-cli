from setuptools import setup

setup(
    name="tokenvisor-cli",
    version="0.1.0",
    packages=["tokenvisor_cli"],
    install_requires=[
        # List your dependencies here
        "pyyaml>=6.0.2",
        "requests>=2.32.0",
        "click>=8.0.0",  # Example if using Click
    ],
    entry_points={
        "console_scripts": [
            "tokenvisor-cli=tokenvisor_cli.main:cli",
        ],
    },
    author="Embedded LLM",
    description="Tokenvisor CLI to deploy & server LLM.",
    license="MIT",
)
