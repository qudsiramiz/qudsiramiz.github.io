#!/home/cephadrius/dlx_venv/bin/python
# -*- coding: utf-8 -*-

import importlib
import os
import time

import dxl_dwld_upld_dscovr_30days_automated as dxl_dwld
import dxl_mp4_dscovr as dxl_mp4

importlib.reload(dxl_dwld)
importlib.reload(dxl_mp4)

time_code_start = time.time()
number_of_days = 365
dxl_dwld.plot_figures_dsco_30days(number_of_days=number_of_days)
dxl_mp4.make_gifs(number_of_days=number_of_days)
#plt.close("all")
# Copy the gif file to google drive
os.system("cp /home/cephadrius/Desktop/git/qudsiramiz.github.io/images/moving_pictures/* /home/cephadrius/google-drive/Studies/Research/bu_research/dxl/figures/vids/")
print(f"Time taken: {round(time.time() - time_code_start, 2)} seconds")
