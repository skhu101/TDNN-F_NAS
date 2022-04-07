import sys

flag = 0
dir = sys.argv[1]
use_gumbel = sys.argv[2]
flops_coef = float(sys.argv[3])
network_type = sys.argv[4]

if network_type == 'tdnn':
    count = 2
elif network_type == 'cnn-tdnn':
    count = 7


with open(dir+'/change.config', 'w') as out_f:
    while count <= 15:
        tdnnf_name = 'tdnnf'+str(count)
        if network_type == 'tdnn':
            out_f.write("component name="+tdnnf_name+".alpha type=ConstantFunctionComponent input-dim=220 output-dim=8 is-updatable=true use-natural-gradient=false"+"\n")
            out_f.write("component-node name="+tdnnf_name+".alpha component="+tdnnf_name+".alpha input=lda"+"\n")
        elif network_type == 'cnn-tdnn':
            out_f.write("component name="+tdnnf_name+".alpha type=ConstantFunctionComponent input-dim=40 output-dim=8 is-updatable=true use-natural-gradient=false"+"\n")
            out_f.write("component-node name="+tdnnf_name+".alpha component="+tdnnf_name+".alpha input=input"+"\n")
        if use_gumbel == "true":
            out_f.write("component name="+tdnnf_name+".softmax type=GumbelSoftmaxFlopsComponent dim=8 scale=" + str(flops_coef)+ " temp-proportion=1.0"+"\n")
            out_f.write("component-node name="+tdnnf_name+".softmax component="+tdnnf_name+".softmax input="+tdnnf_name+".alpha"+"\n")
        else:
            out_f.write("component name="+tdnnf_name+".softmax type=SoftmaxFlopsComponent dim=8 scale=" + str(flops_coef)+ "\n")
            out_f.write("component-node name="+tdnnf_name+".softmax component="+tdnnf_name+".softmax input="+tdnnf_name+".alpha"+"\n")
        count += 1
