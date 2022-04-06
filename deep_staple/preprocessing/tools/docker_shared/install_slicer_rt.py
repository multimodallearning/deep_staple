import slicer
import os

ret_val = slicer.app.extensionsManagerModel().installExtension(os.environ['SLICER_RT_FILE'])
slicer.app.exit(ret_val)