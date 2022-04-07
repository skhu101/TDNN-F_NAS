import sys
import torch
import torch.nn.functional as F

parent_path = sys.argv[1]
child_type = sys.argv[2]
network_type = sys.argv[3]
dim_num = 8

count = 0
prob_tensor = torch.zeros(14,dim_num)

with open(parent_path+'/final_txt.mdl','r') as f:
    for line in f.readlines():
        if "alpha <ConstantFunctionComponent>" in line:
            for j in range(dim_num):
                prob_tensor[count,j] = float(line.split('[')[1].split(' ')[j+1])
            count += 1   
#with open(parent_path+'/log/train.98.1.log','r') as f:
#    for line in reversed(f.readlines()):
#        line = line.strip()
#        if count <= 13 and 'log_alpha' in line:
#            for j in range(dim_num):
#                prob_tensor[13-count,j] = float(line.split('[')[1].split(' ')[j+1])
#            count += 1   

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

print(next_top_dct)

context_left_offset=[25,50,80,100,120,160,200, 240]
context_right_offset=[25,50,80,100,120,160,200,240]

if network_type == 'tdnn':
    for top_id in range(5):
        print(next_top_dct[list(next_top_dct.keys())[top_id]])
        context_info = next_top_dct[list(next_top_dct.keys())[top_id]]

        lst = []
        param = 220*1536 + 1536*256+(256*1536+1536*256+256*6008)*2+1536*14
        for time_offsets_num in range(14):
            lst.append(context_left_offset[context_info[time_offsets_num][1]])
            if time_offsets_num == 4:
                param += 1536*int(context_left_offset[context_info[time_offsets_num][1]])*2
            else:
                param += 1536*int(context_left_offset[context_info[time_offsets_num][1]])*2*2

        print('top'+ str(top_id) + ' ',lst, str(param/1000000)+'M')
        with open(parent_path+'/configs/arch.txt', 'a+') as f:
            f.write('top'+ str(top_id) + ' ')
            for item in lst:
                f.write(str(item)+' ')
            f.write(str(param/1000000)+'M\n')
        
        with open(parent_path+'/configs/arch.txt', 'a+') as f:
            f.write('top'+ str(top_id) + ' ')
            for item in lst:
                f.write(str(context_left_offset.index(item))+' ')
            f.write(str(param/1000000)+'M\n')
elif network_type == 'cnn-tdnn':
    for top_id in range(5):
        print(next_top_dct[list(next_top_dct.keys())[top_id]])
        context_info = next_top_dct[list(next_top_dct.keys())[top_id]]

        lst = []
        param = 6300000
        for time_offsets_num in range(9):
            lst.append(context_left_offset[context_info[time_offsets_num][1]])
            if time_offsets_num == 0:
                param += 2560*int(context_left_offset[context_info[time_offsets_num][1]])*2+1536*int(context_left_offset[context_info[time_offsets_num][1]])*2+1536*2
            else:
                param += 1536*int(context_left_offset[context_info[time_offsets_num][1]])*2*2+1536*2

        print('top'+ str(top_id) + ' ',lst, str(param/1000000)+'M')
        with open(parent_path+'/configs/arch.txt', 'a+') as f:
            f.write('top'+ str(top_id) + ' ')
            for item in lst:
                f.write(str(item)+' ')
            f.write(str(param/1000000)+'M\n')
        
        with open(parent_path+'/configs/arch.txt', 'a+') as f:
            f.write('top'+ str(top_id) + ' ')
            for item in lst:
                f.write(str(context_left_offset.index(item))+' ')
            f.write(str(param/1000000)+'M\n')
