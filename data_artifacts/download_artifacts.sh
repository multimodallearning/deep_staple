#!/usr/bin/env bash

THIS_SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]:-$0}"; )" &> /dev/null && pwd 2> /dev/null; )";

curl -J https://cloud.imi.uni-luebeck.de/s/9DCkYsmdCfGLP33/download/data_artifacts.zip -o $THIS_SCRIPT_DIR/data_artifacts.zip
unzip $THIS_SCRIPT_DIR/data_artifacts.zip -d $THIS_SCRIPT_DIR/..