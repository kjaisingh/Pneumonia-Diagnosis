#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 12:29:12 29020

@author: jaisi8631
"""

# imports
import kaggle
import os
import shutil


# authentication
kaggle.api.authenticate()


# data retrieval
kaggle.api.dataset_download_files('paultimothymooney/chest-xray-pneumonia', unzip=True)
print("Completed data download.")


# data management
shutil.rmtree('chest_xray/__MACOSX')
shutil.rmtree('chest_xray/chest_xray')
os.rename('chest_xray', 'data')
print("Completed data management.")