# Heat Exchanger Documentation

Welcome to the Heat Exchanger Model project! This repository contains documentation and a simple model for heat exchanger design and analysis, designed to be easy to understand and extend.

## Project Overview

This project provides:
- A basic model for heat exchangers
- Documentation written in Markdown, served using [MkDocs](https://www.mkdocs.org/)
- Easy setup for local development and contribution

## Developer Guide

### Prerequisites
- [uv](https://docs.astral.sh/uv/) (a fast Python package manager and environment tool)

#### Install uv
To install `uv`, follow the [official installation guide](https://docs.astral.sh/uv/getting-started/installation/) or use the following command for your platform:

- **Linux / macOS:**
  ```sh
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
- **Windows (PowerShell):**
  ```powershell
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```

### 1. Clone the Repository
```sh
git clone https://github.com/kilianbra/heat_exchanger.git
cd heat_exchanger
```

### 2. Sync Dependencies and Set Up Environment
Use `uv` to automatically create a virtual environment and install all dependencies from `pyproject.toml` and `uv.lock`:
```sh
uv sync
```
- `uv sync` will create a virtual environment (if not already active) and install all dependencies as specified in the lockfile.

### 3. Serve the Documentation Locally
```sh
uv run mkdocs serve
```
This will start a local server (usually at [http://127.0.0.1:8000](http://127.0.0.1:8000)) where you can view and edit the documentation live.

### 4. Build the Documentation (Optional)
To generate a static site:
```sh
uv run mkdocs build
```
The output will be in the `site/` directory.

## Contributing
- Fork the repository and create a new branch for your changes.
- Make your edits and submit a pull request.
- Please ensure your changes are well-documented.

## Project Structure
```text
heat_exchanger/
  docs/
    index.md
    0Dmodel.md
    eps_ntu.md
  mkdocs.yml
  README.md
  pyproject.toml
  uv.lock
```
- `docs/` contains the Markdown documentation files.
- `mkdocs.yml` is the MkDocs configuration file.
- `pyproject.toml` and `uv.lock` define and lock Python dependencies.
- `README.md` is this file.

## License
This project is open source and available under the [MIT License](LICENSE).

---
For any questions or suggestions, please open an issue or contact the developer kpb30@cam.ac.uk
