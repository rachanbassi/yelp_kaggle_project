__author__ = 'alicebenziger'

import json
import pandas as pd
from glob import glob


def convert(json_file):
    ''' Convert a json string to a flat python dictionary
    which can be passed into Pandas. '''
    load_json = json.loads(json_file)
    for k, v in load_json.items():
        if isinstance(v, list):
            load_json[k] = ','.join(v)
        elif isinstance(v, dict):
            for kk, vv in v.items():
                load_json['%s_%s' % (k, kk)] = vv
            del load_json[k]
    return load_json

for json_filename in glob('*.json'):
    csv_filename = '%s.csv' % json_filename[:-5]
    print 'Converting %s to %s' % (json_filename, csv_filename)
    df = pd.DataFrame([convert(line) for line in file(json_filename)])
    df.to_csv(csv_filename, encoding='utf-8', index=False)
