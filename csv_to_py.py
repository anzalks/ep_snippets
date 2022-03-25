#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
from pprint import pprint

class Args: pass
args_ = Args()

def Csv_to_text(csv_file):
    out_f = str(csv_file.absolute()).split('.')[0] + '.py'
    csv_file = str(csv_file.absolute())
    print(out_f, csv_file)
    csv = pd.read_csv(csv_file, header=[0])
    header_py = str("import numpy as np"+"\n"
                    + "import datetime"+"\n"+ "nan = np.nan"+"\n")
    with open(out_f, "w") as output:
        output.write(header_py)
        output.write('\n')
        for row in  csv.iterrows():
            print('*********************')

            var, val,comment = str(row[1][0]), str(row[1][1]), str(row[1][2])
            line = f'{var} = {val} #{comment}'
            print(line)
            output.write(line+'\n')
def List_files(p):
    f_list = []
    f_list=list(p.glob('**/*csv'))
    return f_list

def main(**kwargs):
    p = Path(kwargs['folder_path'])
    f_list = List_files(p)
    for f in f_list:
        Csv_to_text(f)

if __name__  == '__main__':
    import argparse
    # Argument parser.                                                      
    description = '''Convert expt parameter csv to py file.'''
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--folder-path', '-f'
                        , required = False, default
                        ='./', type=str
                        , help = 'path of folder with csv of expt parameters '
                       )
    parser.parse_args(namespace=args_)
    main(**vars(args_))
