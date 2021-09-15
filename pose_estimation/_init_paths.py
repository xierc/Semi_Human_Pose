# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Rongchang Xie (rongchangxie@pku.edu.cn) 
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp
import os
import sys
import subprocess

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


this_dir = osp.dirname(__file__)

lib_path = osp.join(this_dir, '..', 'lib')
add_path(lib_path)

def install(package):
    subprocess.call([sys.executable, "-m", "pip3", "install", package])

try:
    import pycocotools
except :
    print('Installing pycocotools and make...')
    install('pycocotools')
    # os.system("cd " + lib_path + " \n make")
    # os.system("nvidia-smi")
else:
    print('Already have pycocotools')
