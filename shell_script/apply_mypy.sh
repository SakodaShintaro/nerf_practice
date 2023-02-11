#!/bin/bash

set -eux

SCRIPT_PATH=$(readlink -f $0)

SCRIPT_DIR=$(dirname ${SCRIPT_PATH})

TARGET_DIR=${SCRIPT_DIR}/../python/

mypy --strict $(find ${TARGET_DIR} -name "*.py")
