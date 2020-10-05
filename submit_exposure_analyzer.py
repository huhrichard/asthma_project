import sys
import os

suffix = sys.argv[-1]

lsf_str = "#!/bin/bash\n#BSUB -J asthma_project\n#BSUB -P acc_pandeg01a\n#BSUB -q premium\n" \
          "#BSUB -n 4\n#BSUB -W 48:00\n#BSUB -o analyzer_%J.stdout\n#BSUB -eo analyzer_%J.stderr\n" \
          "#BSUB -R rusage[mem=10000]\nmodule purge\n"
python_cmd = "python exposure_analyzer.py {}".format(suffix)

lsf_name = "asthma.lsf"
script = open(lsf_name, 'w')
script.write(lsf_str)
script.write(python_cmd)
script.close()
os.system("bsub < {}".format(lsf_name))
os.remove(lsf_name)