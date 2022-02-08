#!/bin/bash

source /opt/exp_soft/cms/t3/t3setup

declare -i nFiles=$(ls /grid_mnt/data__data.polcms/cms/tarabini/GENPHOTESTPU2_noSmearing/step3/ | wc -l)

for index in $(seq $nFiles); do
  if test -f /grid_mnt/data__data.polcms/cms/tarabini/GENPHOTESTPU2_noSmearing/step3/STEP3_${index}.root; then
    /opt/exp_soft/cms/t3/t3submit batchScript.sh $index
  fi
done
