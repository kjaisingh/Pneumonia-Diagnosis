import os
import shutil

shutil.rmtree('chest_xray/__MACOSX')
shutil.rmtree('chest_xray/chest_xray')
os.rename('chest_xray', 'data')
print("Completed data management.")