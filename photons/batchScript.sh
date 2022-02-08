#!/bin/bash

cp /home/llr/cms/tarabini/ntuplizer/photons/hgc_ntup_gentorchdataset_PU_morefiles_moreinfo_PHO_LC.py .
python3 hgc_ntup_gentorchdataset_PU_morefiles_moreinfo_PHO_LC.py --n $1
