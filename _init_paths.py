# ------------------------------------------------------------------------------
# Corporation. All rights reserved.
#   Licensed under the MIT License.
#  
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp
import sys
import os
import subprocess

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


this_dir = osp.dirname(__file__)

lib_path = osp.join(this_dir, '..', 'lib')
add_path(lib_path)


def install(package):
    subprocess.call([sys.executable, "-m", "pip", "install", package])
    
install('pycocotools')
os.system("cd " + lib_path + " \n make")