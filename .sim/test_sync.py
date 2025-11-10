import sys
sys.path.insert(0, '.')
from enhancer_gui import _sync_and_export_multichannel
print(_sync_and_export_multichannel(['output_audio/test_gui_sync/A.wav','output_audio/test_gui_sync/B.wav'], True))
