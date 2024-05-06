#!/usr/bin/env bash

set -e
set -x

flake8 run.py HakaseAPI HakaseCore
black run.py HakaseAPI HakaseCore --check
isort run.py HakaseAPI HakaseCore --check-only