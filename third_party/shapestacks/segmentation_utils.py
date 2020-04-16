"""
Utility functions to work with segmentation maps (.map files).
"""

import matplotlib.pyplot as plt
import numpy as np


# max. number of labels allowed in a uint8 map
MAX_LABELS = 256

# max. number of labels in VSEG maps, only labels 0-4 used!
VSEG_LABEL_RESOLUTION = 8

# label semantics in VSEG maps
# 0 : background
# 1 : stable base of stack
# 2 : object violating global stack stability
# 3 : object above stability violation / first to fall
# 4 : top of stack


def load_segmap_as_matrix(
    map_path: str,
    label_resolution: int = VSEG_LABEL_RESOLUTION):
  """
  Loads a .map file and returns a matrix of the label values (uint8 between 0
  and 255).

  Args:
    map_path: path to the .map file to load
    label_resolution: max. number of labels used in the map's encoding,
      must be a power of 2

  Returns:
    A np.ndarray of the semantic segmentation labels.
  """
  png_map = plt.imread(map_path)
  label_bin_size = MAX_LABELS // label_resolution
  lbl_map = np.copy(png_map[:, :, 0]) # slice of first image layer
  lbl_map = lbl_map / label_bin_size
  return lbl_map