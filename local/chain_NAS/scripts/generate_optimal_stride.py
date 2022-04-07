import sys
import torch
import torch.nn.functional as F

config_path = sys.argv[1]
context_offset=[]
for i in range(2, 30):
    context_offset.append(int(sys.argv[i]))
print(context_offset)


time_offsets_num = 0

with open(config_path+'/final.config_temp','r') as f:
    with open(config_path+'/final.config','w') as out_f:
        for line in f.readlines():
            line = line.strip()
            new_line = line
            if 'use-bias=true' in line:
                line.replace('use-bias=false','use-bias=true')
                new_line = line
            if 'time-offsets' in line:
                if time_offsets_num % 2 == 0:
                    if context_offset[time_offsets_num] == 0:
                        new_line = line.split('time-offsets')[0] + 'time-offsets=' + '0' +  ' ' + line.split('time-offsets')[1].split(' ')[1]
                    else:
                        new_line = line.split('time-offsets')[0] + 'time-offsets=' + str(context_offset[time_offsets_num]) + ',' + '0' +  ' ' + line.split('time-offsets')[1].split(' ')[1]
                elif time_offsets_num % 2 == 1:
                    if context_offset[time_offsets_num] == 0:
                        new_line = line.split('time-offsets')[0] + 'time-offsets=' + '0'
                    else:
                        new_line = line.split('time-offsets')[0] + 'time-offsets=' + '0' + ',' + str(context_offset[time_offsets_num])
                time_offsets_num += 1
            out_f.write(new_line+'\n')

lst = []
time_offsets_num = 0
with open(config_path+'/ref.config_temp','r') as f:
    with open(config_path+'/ref.config','w') as out_f:
        for line in f.readlines():
            line = line.strip()
            new_line = line
            if 'use-bias=true' in line:
                line.replace('use-bias=false','use-bias=true')
                new_line = line
            if 'time-offsets' in line:
                if time_offsets_num % 2 == 0:
                    if context_offset[time_offsets_num] == 0:
                        new_line = line.split('time-offsets')[0] + 'time-offsets=' + '0' +  ' ' + line.split('time-offsets')[1].split(' ')[1]
                    else:
                        new_line = line.split('time-offsets')[0] + 'time-offsets=' + str(context_offset[time_offsets_num]) + ',' + '0'+  ' ' + line.split('time-offsets')[1].split(' ')[1]
                elif time_offsets_num % 2 == 1:
                    if context_offset[time_offsets_num] == 0:
                        new_line = line.split('time-offsets')[0] + 'time-offsets=' + '0'
                    else:
                        new_line = line.split('time-offsets')[0] + 'time-offsets=' + '0' + ',' + str(context_offset[time_offsets_num])
                time_offsets_num += 1
            out_f.write(new_line+'\n')          


