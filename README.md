# MIPod CLI

MIPod CLI is a command-line interface tool for deploying and serving Large Language Models (LLMs) with AMD GPU Pod scheduling.

## Features

- Schedule AMD GPU Pods for deploying LLMs.
- Manage and monitor your deployments.
- Easy-to-use CLI interface.

## Installation

You can install MIPod CLI using pip:

```bash
pip install git+https://github.com/Hoipang/tokenvisor-cli
```

## Usage

Once installed, you can use the mipod-cli command to interact with the tool. The main entry point for the CLI is defined in the mipod_cli.main module.

To get started, run:

```bash
mipod-cli --help

mipod-cli validate -f config.yaml
mipod-cli deploy -f config.yaml
```

This will display the available commands and options.
