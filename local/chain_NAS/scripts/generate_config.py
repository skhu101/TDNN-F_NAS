import sys
import torch
import torch.nn.functional as F

offset_len = int(sys.argv[1])
config_path = sys.argv[2]

context_left_offset = []
context_right_offset = []

for i in range(offset_len):
    context_right_offset.append(i)

for i in range(-(offset_len-1), 1):
    context_left_offset.append(i)


time_offsets_num = 0

with open(config_path+'final.config_temp','r') as f:
    with open(config_path+'final.config','w') as out_f:
        for line in f.readlines():
            line = line.strip()
            new_line = line
            if 'use-bias=false' in line and "TdnnDARTSV3Component" in line:
                line = line.replace('use-bias=false','use-bias=true')
                new_line = line
            if 'time-offsets' in line and "TdnnDARTSV3Component" in line:
                if time_offsets_num % 2 == 0: 
                    new_line = line.split('time-offsets')[0] + 'time-offsets='
                    for i in range(offset_len):
                        new_line = new_line + str(context_left_offset[i])
                        if i != offset_len-1:
                            new_line = new_line + ','
                    new_line = new_line +  ' ' + line.split('time-offsets')[1].split(' ')[1]
                elif time_offsets_num % 2 == 1:
                    new_line = line.split('time-offsets')[0] + 'time-offsets='
                    for i in range(offset_len):
                        new_line = new_line + str(context_right_offset[i])
                        if i != offset_len-1:
                            new_line = new_line + ','
                time_offsets_num += 1
            out_f.write(new_line+'\n')

time_offsets_num = 0
with open(config_path+'ref.config_temp','r') as f:
    with open(config_path+'ref.config','w') as out_f:
        for line in f.readlines():
            line = line.strip()
            new_line = line
            if 'use-bias=false' in line and "TdnnDARTSV3Component" in line:
                line = line.replace('use-bias=false','use-bias=true')
                new_line = line
            if 'time-offsets' in line and "TdnnDARTSV3Component" in line:
                if time_offsets_num % 2 == 0: 
                    new_line = line.split('time-offsets')[0] + 'time-offsets='
                    for i in range(offset_len):
                        new_line = new_line + str(context_left_offset[i])
                        if i != offset_len-1:
                            new_line = new_line + ','
                    new_line = new_line +  ' ' + line.split('time-offsets')[1].split(' ')[1]                            
                elif time_offsets_num % 2 == 1:
                    new_line = line.split('time-offsets')[0] + 'time-offsets='
                    for i in range(offset_len):
                        new_line = new_line + str(context_right_offset[i])
                        if i != offset_len-1:
                            new_line = new_line + ','
                time_offsets_num += 1
            out_f.write(new_line+'\n')          
