#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 23:24:55 2020

@author: dilsher
"""
#!/usr/bin/python3
from PIL import Image

def blend_image(img1, img2):
    Im = Image.open(img1)
    Im2 = Image.open(img2).convert(Im.mode)
    
    Im2 = Im2.resize(Im.size)
    
    img = Image.blend(Im,Im2,0.15)
    img.save("Blended_image.jpg")
    return img



