import os

n_estimators =[100,50,20]
max_depth =[20,50,10]

for n in n_estimators:
    for m in max_depth:
        os.system(f"python basic_ml.py -n{n} -m{m}")