#!/bin/bash
echo ">>>>>>>>>>>>>>> $1"
echo -n "Python: "
# python3 -c "import pandas as pd, numpy as np; print(pd.read_csv('$1').memory_usage(deep=True).sum())"
python3 -c "import pandas as pd, numpy as np, sys; sys.path += ['/amir_bin'];from type_casting import Type_casting; df=pd.read_csv('$1');Type_casting(id(df)); print(df.memory_usage(deep=True).sum())"
echo -n "R:      "
r -e "print(object.size(read.csv('$1')))" | cut -d ' ' -f1


# import pandas as pd, numpy as np, sys
# file_name = sys.argv[1]
# d = pd.read_csv(file_name)
# total = df.memory_usage(deep=True).sum()


# dic = {}
# l = [int, float, "object"]
# for i in l:
# 	dic[i] = df[df.dtypes[df.dtypes == i].index].memory_usage(deep=True).mean()
# 	dic['total'] = total
# # ['Amir-personal/todo.csv',
# #  'Amir-personal/todo-weekends.csv',
# #  'lfd-projects/Narcos/tbl_entities/after_removing_pvt_etc.csv',
# #  'lfd-projects/Narcos/tbl_entities/Qeuery_cell_column_index.csv',
# #  'lfd-projects/Narcos/tbl_entities/Query_column_index-without-duplacte-observations.csv',
# #  'lfd-projects/Pakistan_economic_survey/LFD-internship/old/working on.csv',
# #  'lfd-projects/Pakistan_economic_survey/LFD-internship/old/tabula-SUR-1 - Copy.csv',
# #  'lfd-projects/Pakistan_economic_survey/LFD-internship/old/tabula-SUR-1.csv',
# #  'lfd-projects/Pakistan_economic_survey/LFD-internship/old/working_on.csv',
# #  'lfd-projects/Pakistan_economic_survey/LFD-internship/Statistical Appendix/completed/SUR_1/SUR-1_cleaned.csv']