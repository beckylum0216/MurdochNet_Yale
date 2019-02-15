import numpy as np
import os
from PIL import Image

import ImageHeader

class Utility(object):

    def readGIFFile(self, filename, imgHdr = ImageHeader.ImageHeader()):
        frame = Image.open(filename).convert('L')
        imgInput = np.true_divide(np.array(frame),255)

        return imgInput

    def readFileLabel(self, dirname):
        filelist = []
        for root, dirs, files in os.walk(dirname):
            for filename in files:
                filelist.append(filename)

        return filelist