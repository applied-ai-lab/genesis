# =========================== A2I Copyright Header ===========================
#
# Copyright (c) 2003-2019 University of Oxford. All rights reserved.
# Authors: Applied AI Lab, Oxford Robotics Institute, University of Oxford
#          https://ori.ox.ac.uk/labs/a2i/
#
# This file is the property of the University of Oxford.
# Redistribution and use in source and binary forms, with or without
# modification, is not permitted without an explicit licensing agreement
# (research or commercial). No warranty, explicit or implicit, provided.
#
# =========================== A2I Copyright Header ===========================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import NoNorm

def convert_to_np_im(torch_tensor, batch_idx=0):
    return np.moveaxis(torch_tensor.data.numpy()[batch_idx], 0, -1)

def plot(axes, ax1, ax2, tensor=None, title=None, grey=False, axis=False):
    if tensor is not None:
        im = convert_to_np_im(tensor)
        if grey:
            im = im[:, :, 0]
            axes[ax1, ax2].imshow(im, norm=NoNorm(), cmap='gray')
        else:
            axes[ax1, ax2].imshow(im)
    if not axis:
        axes[ax1, ax2].axis('off')
    else:
        axes[ax1, ax2].set_xticks([])
        axes[ax1, ax2].set_yticks([])
    if title is not None:
        axes[ax1, ax2].set_title(title, fontsize=4)
    # axes[ax1, ax2].set_aspect('equal')
