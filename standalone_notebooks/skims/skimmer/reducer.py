import os
import awkward as ak
import math
import json
import subprocess

def parq_reduce(dir_name):
    a = ak.from_parquet(f'{dir_name}/*')
    size = len(a)
    one = math.floor(size / 6) * 1
    two = math.floor(size / 6) * 2
    three = math.floor(size / 6) * 3
    four = math.floor(size / 6) * 4
    five = math.floor(size / 6) * 5
    c = a[:one]
    d = a[one:two]
    e = a[two:three]
    f = a[three:four]
    g = a[four:five]
    h = a[five:]
    del(a)
    ak.to_parquet(c, f'{dir_name}/keep0.parquet')
    ak.to_parquet(d, f'{dir_name}/keep1.parquet')
    ak.to_parquet(e, f'{dir_name}/keep2.parquet')
    ak.to_parquet(f, f'{dir_name}/keep3.parquet')
    ak.to_parquet(g, f'{dir_name}/keep4.parquet')
    ak.to_parquet(h, f'{dir_name}/keep5.parquet')
    subprocess.run(f'find {dir_name} -name "part*.parquet" -exec rm {{}} +', shell=True, check=True)

path = "/project01/ndcms/cmoore24/skims/full_skims/nolepton/data"
with open('to_reduce.json', 'r') as f:
    batch = json.load(f)
#batch = ['JetHT_Run2017B_220701_194050', 'JetHT_Run2017B_240313_161752']
for i in batch:
    print(f'Reducing {i}')
    parq_reduce(f'{path}/{i}')