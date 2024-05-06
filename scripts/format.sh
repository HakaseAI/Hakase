#!/bin/sh -e
set -x

isort run.py HakaseAPI HakaseCore --force-single-line-imports
autoflake --remove-all-unused-imports --recursive --remove-unused-variables run.py HakaseAPI HakaseCore --exclude=__init__.py
black run.py HakaseAPI HakaseCore
isort run.py HakaseAPI HakaseCore