#!/bin/bash
/opt/slicer/Slicer --python-script /tmp/shared/install_slicer_rt.py
/opt/slicer/Slicer --python-script /tmp/shared/data_conversion.py --input-folder /tmp/shared_input --output-folder /tmp/shared_output --export_all_structures
sudo kill -s SIGTERM 1