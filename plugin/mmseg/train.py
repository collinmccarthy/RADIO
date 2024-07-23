import sys
from pathlib import Path

# Hack: add top-level module to path until setup.py exists or PYTHON_PATH has been updated
# Need to add path one level above 'RADIO' project dir to reference correct mmseg path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))  # One above `RADIO` project dir

from radio.plugin.mmseg.mmdet_radio import *  # Register MMDetRADIO
from radio.mmseg.train import main

if __name__ == "__main__":
    main()
