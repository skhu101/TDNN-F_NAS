import sys
import torch
import torch.nn.functional as F

parent_path = sys.argv[1]
child_type = sys.argv[2]
top_id = int(sys.argv[3])
config_path = sys.argv[4]
offset = int(sys.argv[5])
network_type = sys.argv[6]

count = 0

if network_type == 'tdnn':
    prob_tensor = torch.zeros(14,offset)
elif network_type == 'cnn-tdnn':
    prob_tensor = torch.zeros(9,offset)

dim_num = offset

with open(parent_path+'/final_txt.mdl','r') as f:
    for line in f.readlines():
        if "alpha <ConstantFunctionComponent>" in line:
            for j in range(dim_num):
                prob_tensor[count,j] = float(line.split('[')[1].split(' ')[j+1])
            count += 1   

if child_type == 'top':
    prob_tensor = F.softmax(prob_tensor)
elif child_type == 'last':
    prob_tensor = F.softmax(-prob_tensor)
else:
    print('Error: no such type!!')

top_dct = {}
next_top_dct={}

for i in range(prob_tensor.size(0)):
    top_dct = {}
    for j in range(prob_tensor.size(1)):
        if i == 0:
            next_top_dct[prob_tensor[i,j].item()] = [[i,j]]
        else:
            for key in next_top_dct.keys():
                path = next_top_dct[key].copy()
                path.append([i,j])
                top_dct[key*prob_tensor[i,j].item()] = path
    if i >=1:
        next_top_dct={}
        top_list = []
        for key in top_dct.keys():
            top_list.append(key)
        top_list.sort(reverse=True)
        for key in top_list[0:10]:
            next_top_dct[key] = top_dct[key]

# print(next_top_dct)
print(next_top_dct[list(next_top_dct.keys())[top_id-1]])
context_info = next_top_dct[list(next_top_dct.keys())[top_id-1]]

context_left_offset=[25,50,80,100,120,160,200, 240]
context_right_offset=[25,50,80,100,120,160,200,240]
#for i in range(-(offset-1), 1):
#    context_left_offset.append(i)

#for i in range(0, offset):
#    context_right_offset.append(i)


time_offsets_num = 0

with open(config_path+'final.config_temp','r') as f:
    with open(config_path+'final.config','w') as out_f:
        for line in f.readlines():
            line = line.strip()
            new_line = line
            if 'use-bias=true' in line and 'TdnnComponent' in line:
                line.replace('use-bias=false','use-bias=true')
                new_line = line
            if 'time-offsets' in line and 'TdnnComponent' in line:
                if time_offsets_num % 2 == 0:
                    new_line = line.split('output-dim=')[0] + 'output-dim=' + str(context_left_offset[context_info[time_offsets_num//2][1]]) + ' l2-regularize' + line.split('l2-regularize')[1]
                elif time_offsets_num % 2 == 1:
                    new_line = line.split('input-dim=')[0] + 'input-dim=' + str(context_right_offset[context_info[time_offsets_num//2][1]]) + ' output-dim' + line.split('output-dim')[1]
                time_offsets_num += 1
            out_f.write(new_line+'\n')

lst = []
time_offsets_num = 0
with open(config_path+'ref.config_temp','r') as f:
    with open(config_path+'ref.config','w') as out_f:
        for line in f.readlines():
            line = line.strip()
            new_line = line
            if 'use-bias=true' in line and 'TdnnComponent' in line:
                line.replace('use-bias=false','use-bias=true')
                new_line = line
            if 'time-offsets' in line and 'TdnnComponent' in line:
                if time_offsets_num % 2 == 0:
                    lst.append(context_left_offset[context_info[time_offsets_num//2][1]])
                    new_line = line.split('output-dim=')[0] + 'output-dim=' + str(context_left_offset[context_info[time_offsets_num//2][1]]) + ' l2-regularize' + line.split('l2-regularize')[1]
                elif time_offsets_num % 2 == 1:
                    #lst.append(context_right_offset[context_info[time_offsets_num//2][1]])
                    new_line = line.split('input-dim=')[0] + 'input-dim=' + str(context_right_offset[context_info[time_offsets_num//2][1]]) + ' output-dim' + line.split('output-dim')[1]
                time_offsets_num += 1
            out_f.write(new_line+'\n')          

print(child_type,lst)
with open(config_path+'arch.txt', 'w') as f:
    for item in lst:
        f.write(str(item)+' ')
    f.write('\n') 
    for item in lst:
        f.write(str(context_left_offset.index(item))+' ')
   

