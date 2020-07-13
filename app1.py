# -*- coding: utf-8 -*-

import functools
import os
import os.path as osp
import re
import webbrowser


from labelme.label_file import LabelFile



def saveLabels(filename,imagename,imageData,shapes):
    lf = LabelFile()
    lf.save(
                filename=filename,
                shapes=shapes,
                imagePath=imagename,
                imageData=imageData,
                imageHeight=1200,
                imageWidth=1920,
                otherData=None,
                flags={},
            )

shapes=[{'label': 'cat', 'points': [(1093.25, 425.25), (1258.25, 571.5)], 'group_id': None, 'shape_type': 'rectangle',
      'flags': {}}]
filename='./1/11.json'
imagename='11.jpg'
imageData=LabelFile.load_image_file('./1/11.jpg')
saveLabels(filename,imagename,imageData,shapes)


