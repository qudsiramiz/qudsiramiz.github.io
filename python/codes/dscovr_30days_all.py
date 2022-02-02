#!/home/cephadrius/dlx_venv/bin/python
# -*- coding: utf-8 -*-

import os
import time
from dxl_dwld_upld_dscovr_30days_automated import plot_figures_dsco_30days
from dxl_mp4_dscovr import make_gifs

time_code_start = time.time()
number_of_days = 365
plot_figures_dsco_30days(number_of_days=number_of_days)
make_gifs(number_of_days=number_of_days)

# Copy the gif file to google drive
os.system("cp /home/cephadrius/Desktop/git/qudsiramiz.github.io/images/moving_pictures/* /home/cephadrius/google-drive/Studies/Research/bu_research/dxl/figures/vids/")
print(f"Time taken: {round(time.time() - time_code_start, 2)} seconds")