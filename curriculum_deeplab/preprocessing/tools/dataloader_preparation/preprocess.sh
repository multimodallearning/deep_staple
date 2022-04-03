#!/bin/zsh

TCIA_INPUT="/Users/christianweihsbach/tcia_mangling_tmp/tcia_crossmoda_001_250/manifest-1614264588831/Vestibular-Schwannoma-SEG"
TCIA_CONVENIENT="/Users/christianweihsbach/tcia_mangling_tmp/tcia_crossmoda_001_250_convenient"
# python3 preprocessing/TCIA_data_convert_into_convenient_folder_structure.py --input $TCIA_INPUT --output $TCIA_CONVENIENT

SLICER_PATH="/Applications/Slicer.app/Contents/MacOS/Slicer"
PREPRO_SCRIPT_PATH="/Users/christianweihsbach/code/VS_Seg/preprocessing/data_conversion.py"
INPUT_PATH=$TCIA_CONVENIENT
OUTPUT_PATH="/Users/christianweihsbach/tcia_mangling_tmp/tcia_crossmoda_001_250_converted_reg_new"
# $SLICER_PATH --python-script $PREPRO_SCRIPT_PATH --input-folder $INPUT_PATH --output-folder $OUTPUT_PATH --export_all_structures

SLICER_PATH_ALT="/Applications/Slicer.app/Contents/bin/PythonSlicer"
$SLICER_PATH_ALT $SLICER_PATH --launch $SLICER_PATH_ALT $PREPRO_SCRIPT_PATH --input-folder $INPUT_PATH --output-folder $OUTPUT_PATH --export_all_structures