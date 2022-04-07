import sys
import torch
import torch.nn.functional as F

parent_path = sys.argv[1]
child_type = sys.argv[2]
top_id = int(sys.argv[3])
config_path = sys.argv[4]
offset = int(sys.argv[5])
model_type = sys.argv[6]

if model_type == "tdnn":
    prob_tensor = torch.zeros(28,offset)
elif model_type == "cnn-tdnn":
    prob_tensor = torch.zeros(18,offset)
count = 1

# with open('final_txt.mdl','r') as f:
if model_type == "tdnn":
    with open(parent_path+'/final_txt.mdl','r') as f:
        for line in f.readlines():
            line = line.strip()
            if '<BiasParams>' in line:
                if count > 2 and count < 31:
                    prob = line.split('[')[1].split(' ')[1:offset+1]   
                    prob_list = [float(item) for item in prob]
                    prob_tensor[count-3,:] = torch.Tensor(prob_list)
                count += 1    
elif model_type == "cnn-tdnn":
    with open(parent_path+'/final_txt.mdl','r') as f:
        for line in f.readlines():
            line = line.strip()
            if '<BiasParams>' in line:
                if count > 7 and count < 26:
                    prob = line.split('[')[1].split(' ')[1:offset+1]   
                    prob_list = [float(item) for item in prob]
                    prob_tensor[count-8,:] = torch.Tensor(prob_list)
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

context_left_offset=[]
context_right_offset=[]
for i in range(-(offset-1), 1):
    context_left_offset.append(i)

for i in range(0, offset):
    context_right_offset.append(i)

# lst = []
# time_offsets_num = 0
# for i in range(len(context_info)):
#     if time_offsets_num % 2 == 0:
#         lst.append(context_left_offset[context_info[time_offsets_num][1]])
#     else:
#         lst.append(context_right_offset[context_info[time_offsets_num][1]])
#     time_offsets_num += 1
# print(lst)


# context_left_offset = [-3,-2,-1,0]
# context_right_offset = [0,1,2,3]

time_offsets_num = 0

with open(config_path+'final.config_temp','r') as f:
    with open(config_path+'final.config','w') as out_f:
        for line in f.readlines():
            line = line.strip()
            new_line = line
            if 'use-bias=true' in line:
                line.replace('use-bias=false','use-bias=true')
                new_line = line
            if 'time-offsets' in line and "TdnnComponent" in line:
                if time_offsets_num % 2 == 0:
                    if context_info[time_offsets_num][1] == offset-1:
                        new_line = line.split('time-offsets')[0] + 'time-offsets=' + '0' +  ' ' + line.split('time-offsets')[1].split(' ')[1]
                    else:
                        new_line = line.split('time-offsets')[0] + 'time-offsets=' + str(context_left_offset[context_info[time_offsets_num][1]]) + ',' + '0' +  ' ' + line.split('time-offsets')[1].split(' ')[1]
                elif time_offsets_num % 2 == 1:
                    if context_info[time_offsets_num][1] == 0:
                        new_line = line.split('time-offsets')[0] + 'time-offsets=' + '0'
                    else:
                        new_line = line.split('time-offsets')[0] + 'time-offsets=' + '0' + ',' + str(context_right_offset[context_info[time_offsets_num][1]])
                time_offsets_num += 1
            out_f.write(new_line+'\n')

lst = []
time_offsets_num = 0
with open(config_path+'ref.config_temp','r') as f:
    with open(config_path+'ref.config','w') as out_f:
        for line in f.readlines():
            line = line.strip()
            new_line = line
            if 'use-bias=true' in line:
                line.replace('use-bias=false','use-bias=true')
                new_line = line
            if 'time-offsets' in line and "TdnnComponent" in line:
                if time_offsets_num % 2 == 0:
                    lst.append(context_left_offset[context_info[time_offsets_num][1]])
                    if context_info[time_offsets_num][1] == offset-1:
                        new_line = line.split('time-offsets')[0] + 'time-offsets=' + '0' +  ' ' + line.split('time-offsets')[1].split(' ')[1]
                    else:
                        new_line = line.split('time-offsets')[0] + 'time-offsets=' + str(context_left_offset[context_info[time_offsets_num][1]]) + ',' + '0'+  ' ' + line.split('time-offsets')[1].split(' ')[1]
                elif time_offsets_num % 2 == 1:
                    lst.append(context_right_offset[context_info[time_offsets_num][1]])
                    if context_info[time_offsets_num][1] == 0:
                        new_line = line.split('time-offsets')[0] + 'time-offsets=' + '0'
                    else:
                        new_line = line.split('time-offsets')[0] + 'time-offsets=' + '0' + ',' + str(context_right_offset[context_info[time_offsets_num][1]])
                time_offsets_num += 1
            out_f.write(new_line+'\n')          

print(child_type,lst)
with open(config_path+'arch.txt', 'w') as f:
    for item in lst:
        f.write(str(item)+' ')

