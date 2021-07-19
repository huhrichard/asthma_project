import sys
import os

suffix = sys.argv[-1]
outcome_binary_dict = {
    # 'asthma': True,
    # 'asthma(act_score)':False,
    # 'age_greaterthan5_diagnosed_asthma': True,
    # 'age_diagnosed_asthma': False,
    'daily_controller_past6months': True,
    'emergency_dept': True,
    # 'emergency_dept_pastyr_count': False,
    'hospitalize_overnight': True,
    # 'hospitalize_overnight_pastyr_count': False,
    # 'regular_asthma_symptoms_past6months': True,
    # 'regular_asthma_symptoms_daysCount_pastWeek': False
    #  'asthma(act_score_lessthan20)': True,
    #  'emergency_dept_pastyr_count(greaterthan0)': True,
    #  'emergency_dept_pastyr_count(greaterthan_nz_median)': True,
    #  'hospitalize_overnight_pastyr_count(greaterthan0)': True,
    #  'hospitalize_overnight_pastyr_count(greaterthan_nz_median)': True,
    #  'regular_asthma_symptoms_daysCount_pastWeek(greaterthan0)': True,
    # 'regular_asthma_symptoms_daysCount_pastWeek(greaterthan_nz_median)': True
}

for outcome in outcome_binary_dict:
    lsf_str = "#!/bin/bash\n#BSUB -J asthma_project\n#BSUB -P acc_pandeg01a\n#BSUB -q express\n" \
              "#BSUB -n 4\n#BSUB -W 2:00\n#BSUB -o analyzer_%J.stdout\n#BSUB -eo analyzer_%J.stderr\n" \
              "#BSUB -R rusage[mem=10000]\nmodule purge\n"
    outcome_replaced = outcome.replace('(', '\(').replace(')', '\)')
    python_cmd = "python exposure_analyzer.py {} {}".format(outcome_replaced, suffix)

    lsf_name = "{}.lsf".format(outcome)
    lsf_replaced_name = "{}.lsf".format(outcome_replaced)
    script = open(lsf_name, 'w')
    script.write(lsf_str)
    script.write(python_cmd)
    script.close()
    os.system("bsub < {}".format(lsf_replaced_name))
    os.remove(lsf_name)