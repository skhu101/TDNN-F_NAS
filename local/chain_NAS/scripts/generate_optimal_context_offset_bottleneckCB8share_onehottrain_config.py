import sys
config_path = sys.argv[1]
network_type = sys.argv[2]
context_offset=[]
if network_type == 'tdnn':
    for i in range(3, 31):
        context_offset.append(int(sys.argv[i]))
elif network_type == 'cnn-tdnn':
    for i in range(3, 21):
        context_offset.append(int(sys.argv[i]))

print(context_offset)
if network_type == 'tdnn':
    count = 2
    with open(config_path+'/final.config', 'w') as out_f:
        with open(config_path+'/final_ori.config') as f:
            for line in f.readlines():
                line = line.strip()
                if 'component name=tdnnf'+str(count)+'.linear' in line:
                    tdnnf_name = 'tdnnf'+str(count)
                    tdnnfpre_name = 'tdnnf'+str(count-1)
                    out_f.write("component name="+tdnnf_name+".softmax type=OnehotFunctionComponent input-dim=220 output-dim=8 is-updatable=true use-natural-gradient=false"+"\n")
                    out_f.write("component-node name="+tdnnf_name+".softmax component="+tdnnf_name+".softmax input=lda"+"\n")
                    #out_f.write("component name="+tdnnf_name+".softmax type=SoftmaxComponent dim=8"+"\n")
                    #out_f.write("component-node name="+tdnnf_name+".softmax component="+tdnnf_name+".softmax input="+tdnnf_name+".alpha"+"\n")
                    out_f.write("dim-range-node name="+tdnnf_name+".softmax0 input-node="+tdnnf_name+".softmax dim-offset=0 dim=1"+"\n")
                    out_f.write("dim-range-node name="+tdnnf_name+".softmax1 input-node="+tdnnf_name+".softmax dim-offset=1 dim=1"+"\n")
                    out_f.write("dim-range-node name="+tdnnf_name+".softmax2 input-node="+tdnnf_name+".softmax dim-offset=2 dim=1"+"\n")
                    out_f.write("dim-range-node name="+tdnnf_name+".softmax3 input-node="+tdnnf_name+".softmax dim-offset=3 dim=1"+"\n")
                    out_f.write("dim-range-node name="+tdnnf_name+".softmax4 input-node="+tdnnf_name+".softmax dim-offset=4 dim=1"+"\n")
                    out_f.write("dim-range-node name="+tdnnf_name+".softmax5 input-node="+tdnnf_name+".softmax dim-offset=5 dim=1"+"\n")
                    out_f.write("dim-range-node name="+tdnnf_name+".softmax6 input-node="+tdnnf_name+".softmax dim-offset=6 dim=1"+"\n")
                    out_f.write("dim-range-node name="+tdnnf_name+".softmax7 input-node="+tdnnf_name+".softmax dim-offset=7 dim=1"+"\n")

                    out_f.write("component name="+tdnnf_name+"0.copyn type=CopyNComponent input-dim=1 output-dim=25"+"\n")
                    out_f.write("component-node name="+tdnnf_name+"0.copyn component="+tdnnf_name+"0.copyn input=Sum("+tdnnf_name+".softmax0,"+tdnnf_name+".softmax1,"+tdnnf_name+".softmax2,"+tdnnf_name+".softmax3,"+tdnnf_name+".softmax4,"+tdnnf_name+".softmax5,"+tdnnf_name+".softmax6,"+tdnnf_name+".softmax7)"+"\n")
                    out_f.write("component name="+tdnnf_name+"1.copyn type=CopyNComponent input-dim=1 output-dim=25"+"\n")
                    out_f.write("component-node name="+tdnnf_name+"1.copyn component="+tdnnf_name+"1.copyn input=Sum("+tdnnf_name+".softmax1,"+tdnnf_name+".softmax2,"+tdnnf_name+".softmax3,"+tdnnf_name+".softmax4,"+tdnnf_name+".softmax5,"+tdnnf_name+".softmax6,"+tdnnf_name+".softmax7)"+"\n")
                    out_f.write("component name="+tdnnf_name+"2.copyn type=CopyNComponent input-dim=1 output-dim=30"+"\n")
                    out_f.write("component-node name="+tdnnf_name+"2.copyn component="+tdnnf_name+"2.copyn input=Sum("+tdnnf_name+".softmax2,"+tdnnf_name+".softmax3,"+tdnnf_name+".softmax4,"+tdnnf_name+".softmax5,"+tdnnf_name+".softmax6,"+tdnnf_name+".softmax7)"+"\n")
                    out_f.write("component name="+tdnnf_name+"3.copyn type=CopyNComponent input-dim=1 output-dim=20"+"\n")
                    out_f.write("component-node name="+tdnnf_name+"3.copyn component="+tdnnf_name+"3.copyn input=Sum("+tdnnf_name+".softmax3,"+tdnnf_name+".softmax4,"+tdnnf_name+".softmax5,"+tdnnf_name+".softmax6,"+tdnnf_name+".softmax7)"+"\n")
                    out_f.write("component name="+tdnnf_name+"4.copyn type=CopyNComponent input-dim=1 output-dim=20"+"\n")
                    out_f.write("component-node name="+tdnnf_name+"4.copyn component="+tdnnf_name+"4.copyn input=Sum("+tdnnf_name+".softmax4,"+tdnnf_name+".softmax5,"+tdnnf_name+".softmax6,"+tdnnf_name+".softmax7)"+"\n")
                    out_f.write("component name="+tdnnf_name+"5.copyn type=CopyNComponent input-dim=1 output-dim=40"+"\n")
                    out_f.write("component-node name="+tdnnf_name+"5.copyn component="+tdnnf_name+"5.copyn input=Sum("+tdnnf_name+".softmax5,"+tdnnf_name+".softmax6,"+tdnnf_name+".softmax7)"+"\n")
                    out_f.write("component name="+tdnnf_name+"6.copyn type=CopyNComponent input-dim=1 output-dim=40"+"\n")
                    out_f.write("component-node name="+tdnnf_name+"6.copyn component="+tdnnf_name+"6.copyn input=Sum("+tdnnf_name+".softmax6,"+tdnnf_name+".softmax7)"+"\n")
                    out_f.write("component name="+tdnnf_name+"7.copyn type=CopyNComponent input-dim=1 output-dim=40"+"\n")
                    out_f.write("component-node name="+tdnnf_name+"7.copyn component="+tdnnf_name+"7.copyn input="+tdnnf_name+".softmax7"+"\n")

                    if context_offset[(count-2)*2] == 0:
                        out_f.write("component name="+tdnnf_name+".linear type=TdnnComponent input-dim=1536 output-dim=240 l2-regularize=0.01 max-change=0.75 use-bias=false time-offsets=0 orthonormal-constraint=-1.0"+"\n")
                    else:
                        out_f.write("component name="+tdnnf_name+".linear type=TdnnComponent input-dim=1536 output-dim=240 l2-regularize=0.01 max-change=0.75 use-bias=false time-offsets="+str(context_offset[(count-2)*2])+",0 orthonormal-constraint=-1.0"+"\n")
                    if count == 2:
                        out_f.write("component-node name="+tdnnf_name+".linear component="+tdnnf_name+".linear input=tdnn1.dropout"+"\n")
                    else:
                        out_f.write("component-node name="+tdnnf_name+".linear component="+tdnnf_name+".linear input="+tdnnfpre_name+".noop"+"\n")                  

                    out_f.write("dim-range-node name="+tdnnf_name+"0.linear input-node="+tdnnf_name+".linear dim-offset=0 dim=25"+"\n")
                    out_f.write("dim-range-node name="+tdnnf_name+"1.linear input-node="+tdnnf_name+".linear dim-offset=25 dim=25"+"\n")
                    out_f.write("dim-range-node name="+tdnnf_name+"2.linear input-node="+tdnnf_name+".linear dim-offset=50 dim=30"+"\n")
                    out_f.write("dim-range-node name="+tdnnf_name+"3.linear input-node="+tdnnf_name+".linear dim-offset=80 dim=20"+"\n") 
                    out_f.write("dim-range-node name="+tdnnf_name+"4.linear input-node="+tdnnf_name+".linear dim-offset=100 dim=20"+"\n")
                    out_f.write("dim-range-node name="+tdnnf_name+"5.linear input-node="+tdnnf_name+".linear dim-offset=120 dim=40"+"\n")
                    out_f.write("dim-range-node name="+tdnnf_name+"6.linear input-node="+tdnnf_name+".linear dim-offset=160 dim=40"+"\n")
                    out_f.write("dim-range-node name="+tdnnf_name+"7.linear input-node="+tdnnf_name+".linear dim-offset=200 dim=40"+"\n")

                    out_f.write("component name="+tdnnf_name+"0.output type=ElementwiseProductComponent input-dim=50 output-dim=25"+"\n")
                    out_f.write("component-node name="+tdnnf_name+"0.output component="+tdnnf_name+"0.output input=Append("+tdnnf_name+"0.copyn, "+tdnnf_name+"0.linear)"+"\n")
                    out_f.write("component name="+tdnnf_name+"1.output type=ElementwiseProductComponent input-dim=50 output-dim=25"+"\n")
                    out_f.write("component-node name="+tdnnf_name+"1.output component="+tdnnf_name+"1.output input=Append("+tdnnf_name+"1.copyn, "+tdnnf_name+"1.linear)"+"\n")
                    out_f.write("component name="+tdnnf_name+"2.output type=ElementwiseProductComponent input-dim=60 output-dim=30"+"\n")
                    out_f.write("component-node name="+tdnnf_name+"2.output component="+tdnnf_name+"2.output input=Append("+tdnnf_name+"2.copyn, "+tdnnf_name+"2.linear)"+"\n")
                    out_f.write("component name="+tdnnf_name+"3.output type=ElementwiseProductComponent input-dim=40 output-dim=20"+"\n")
                    out_f.write("component-node name="+tdnnf_name+"3.output component="+tdnnf_name+"3.output input=Append("+tdnnf_name+"3.copyn, "+tdnnf_name+"3.linear)"+"\n") 
                    out_f.write("component name="+tdnnf_name+"4.output type=ElementwiseProductComponent input-dim=40 output-dim=20"+"\n")
                    out_f.write("component-node name="+tdnnf_name+"4.output component="+tdnnf_name+"4.output input=Append("+tdnnf_name+"4.copyn, "+tdnnf_name+"4.linear)"+"\n") 
                    out_f.write("component name="+tdnnf_name+"5.output type=ElementwiseProductComponent input-dim=80 output-dim=40"+"\n")
                    out_f.write("component-node name="+tdnnf_name+"5.output component="+tdnnf_name+"5.output input=Append("+tdnnf_name+"5.copyn, "+tdnnf_name+"5.linear)"+"\n") 
                    out_f.write("component name="+tdnnf_name+"6.output type=ElementwiseProductComponent input-dim=80 output-dim=40"+"\n")
                    out_f.write("component-node name="+tdnnf_name+"6.output component="+tdnnf_name+"6.output input=Append("+tdnnf_name+"6.copyn, "+tdnnf_name+"6.linear)"+"\n") 
                    out_f.write("component name="+tdnnf_name+"7.output type=ElementwiseProductComponent input-dim=80 output-dim=40"+"\n")
                    out_f.write("component-node name="+tdnnf_name+"7.output component="+tdnnf_name+"7.output input=Append("+tdnnf_name+"7.copyn, "+tdnnf_name+"7.linear)"+"\n") 
                    if context_offset[(count-2)*2+1] == 0:
                        out_f.write("component name="+tdnnf_name+".affine type=TdnnComponent input-dim=240 output-dim=1536 l2-regularize=0.01 max-change=0.75 time-offsets=0"+"\n")
                    else:
                        out_f.write("component name="+tdnnf_name+".affine type=TdnnComponent input-dim=240 output-dim=1536 l2-regularize=0.01 max-change=0.75 time-offsets=0,"+str(context_offset[(count-2)*2+1])+"\n")
                    out_f.write("component-node name="+tdnnf_name+".affine component="+tdnnf_name+".affine input=Append("+tdnnf_name+"0.output,"+tdnnf_name+"1.output,"+tdnnf_name+"2.output,"+tdnnf_name+"3.output,"+tdnnf_name+"4.output,"+tdnnf_name+"5.output,"+tdnnf_name+"6.output,"+tdnnf_name+"7.output)"+"\n")
                        
                    out_f.write("component name="+tdnnf_name+".relu type=RectifiedLinearComponent dim=1536 self-repair-scale=1e-05"+"\n")
                    out_f.write("component-node name="+tdnnf_name+".relu component="+tdnnf_name+".relu input="+tdnnf_name+".affine"+"\n")
                    out_f.write("component name="+tdnnf_name+".batchnorm type=BatchNormComponent dim=1536"+"\n")
                    out_f.write("component-node name="+tdnnf_name+".batchnorm component="+tdnnf_name+".batchnorm input="+tdnnf_name+".relu"+"\n")                
                elif 'component-node name=tdnnf'+str(count)+'.linear' in line or 'component name=tdnnf'+str(count)+'.affine' in line or 'component-node name=tdnnf'+str(count)+'.affine' in line:
                    continue
                elif 'component-node name=tdnnf'+str(count)+'.relu' in line or 'component name=tdnnf'+str(count)+'.relu' in line:
                    continue
                elif 'component-node name=tdnnf'+str(count)+'.batchnorm' in line or 'component name=tdnnf'+str(count)+'.batchnorm' in line:
                    continue    
                elif 'component name=tdnnf'+str(count)+'.dropout' in line:
                    continue              
                elif 'component-node name=tdnnf'+str(count)+'.dropout' in line:
                    out_f.write("component name="+tdnnf_name+".dropout type=GeneralDropoutComponent dim=1536 dropout-proportion=0.0 continuous=true"+"\n")
                    out_f.write("component-node name="+tdnnf_name+".dropout component="+tdnnf_name+".dropout input="+tdnnf_name+".batchnorm"+"\n")  
                    count += 1
                else:
                    out_f.write(line+'\n')


    count = 2
    with open(config_path+'/ref.config', 'w') as out_f:
        with open(config_path+'/ref_ori.config') as f:
            for line in f.readlines():
                line = line.strip()
                if 'component name=tdnnf'+str(count)+'.linear' in line:
                    tdnnf_name = 'tdnnf'+str(count)
                    tdnnfpre_name = 'tdnnf'+str(count-1)
                    out_f.write("component name="+tdnnf_name+".softmax type=OnehotFunctionComponent input-dim=220 output-dim=8 is-updatable=true use-natural-gradient=false"+"\n")
                    out_f.write("component-node name="+tdnnf_name+".softmax component="+tdnnf_name+".softmax input=lda"+"\n")
                    #out_f.write("component name="+tdnnf_name+".softmax type=SoftmaxComponent dim=8"+"\n")
                    #out_f.write("component-node name="+tdnnf_name+".softmax component="+tdnnf_name+".softmax input="+tdnnf_name+".alpha"+"\n")
                    out_f.write("dim-range-node name="+tdnnf_name+".softmax0 input-node="+tdnnf_name+".softmax dim-offset=0 dim=1"+"\n")
                    out_f.write("dim-range-node name="+tdnnf_name+".softmax1 input-node="+tdnnf_name+".softmax dim-offset=1 dim=1"+"\n")
                    out_f.write("dim-range-node name="+tdnnf_name+".softmax2 input-node="+tdnnf_name+".softmax dim-offset=2 dim=1"+"\n")
                    out_f.write("dim-range-node name="+tdnnf_name+".softmax3 input-node="+tdnnf_name+".softmax dim-offset=3 dim=1"+"\n")
                    out_f.write("dim-range-node name="+tdnnf_name+".softmax4 input-node="+tdnnf_name+".softmax dim-offset=4 dim=1"+"\n")
                    out_f.write("dim-range-node name="+tdnnf_name+".softmax5 input-node="+tdnnf_name+".softmax dim-offset=5 dim=1"+"\n")
                    out_f.write("dim-range-node name="+tdnnf_name+".softmax6 input-node="+tdnnf_name+".softmax dim-offset=6 dim=1"+"\n")
                    out_f.write("dim-range-node name="+tdnnf_name+".softmax7 input-node="+tdnnf_name+".softmax dim-offset=7 dim=1"+"\n")

                    out_f.write("component name="+tdnnf_name+"0.copyn type=CopyNComponent input-dim=1 output-dim=25"+"\n")
                    out_f.write("component-node name="+tdnnf_name+"0.copyn component="+tdnnf_name+"0.copyn input=Sum("+tdnnf_name+".softmax0,"+tdnnf_name+".softmax1,"+tdnnf_name+".softmax2,"+tdnnf_name+".softmax3,"+tdnnf_name+".softmax4,"+tdnnf_name+".softmax5,"+tdnnf_name+".softmax6,"+tdnnf_name+".softmax7)"+"\n")
                    out_f.write("component name="+tdnnf_name+"1.copyn type=CopyNComponent input-dim=1 output-dim=25"+"\n")
                    out_f.write("component-node name="+tdnnf_name+"1.copyn component="+tdnnf_name+"1.copyn input=Sum("+tdnnf_name+".softmax1,"+tdnnf_name+".softmax2,"+tdnnf_name+".softmax3,"+tdnnf_name+".softmax4,"+tdnnf_name+".softmax5,"+tdnnf_name+".softmax6,"+tdnnf_name+".softmax7)"+"\n")
                    out_f.write("component name="+tdnnf_name+"2.copyn type=CopyNComponent input-dim=1 output-dim=30"+"\n")
                    out_f.write("component-node name="+tdnnf_name+"2.copyn component="+tdnnf_name+"2.copyn input=Sum("+tdnnf_name+".softmax2,"+tdnnf_name+".softmax3,"+tdnnf_name+".softmax4,"+tdnnf_name+".softmax5,"+tdnnf_name+".softmax6,"+tdnnf_name+".softmax7)"+"\n")
                    out_f.write("component name="+tdnnf_name+"3.copyn type=CopyNComponent input-dim=1 output-dim=20"+"\n")
                    out_f.write("component-node name="+tdnnf_name+"3.copyn component="+tdnnf_name+"3.copyn input=Sum("+tdnnf_name+".softmax3,"+tdnnf_name+".softmax4,"+tdnnf_name+".softmax5,"+tdnnf_name+".softmax6,"+tdnnf_name+".softmax7)"+"\n")
                    out_f.write("component name="+tdnnf_name+"4.copyn type=CopyNComponent input-dim=1 output-dim=20"+"\n")
                    out_f.write("component-node name="+tdnnf_name+"4.copyn component="+tdnnf_name+"4.copyn input=Sum("+tdnnf_name+".softmax4,"+tdnnf_name+".softmax5,"+tdnnf_name+".softmax6,"+tdnnf_name+".softmax7)"+"\n")
                    out_f.write("component name="+tdnnf_name+"5.copyn type=CopyNComponent input-dim=1 output-dim=40"+"\n")
                    out_f.write("component-node name="+tdnnf_name+"5.copyn component="+tdnnf_name+"5.copyn input=Sum("+tdnnf_name+".softmax5,"+tdnnf_name+".softmax6,"+tdnnf_name+".softmax7)"+"\n")
                    out_f.write("component name="+tdnnf_name+"6.copyn type=CopyNComponent input-dim=1 output-dim=40"+"\n")
                    out_f.write("component-node name="+tdnnf_name+"6.copyn component="+tdnnf_name+"6.copyn input=Sum("+tdnnf_name+".softmax6,"+tdnnf_name+".softmax7)"+"\n")
                    out_f.write("component name="+tdnnf_name+"7.copyn type=CopyNComponent input-dim=1 output-dim=40"+"\n")
                    out_f.write("component-node name="+tdnnf_name+"7.copyn component="+tdnnf_name+"7.copyn input="+tdnnf_name+".softmax7"+"\n")

                    if context_offset[(count-2)*2] == 0:
                        out_f.write("component name="+tdnnf_name+".linear type=TdnnComponent input-dim=1536 output-dim=240 l2-regularize=0.01 max-change=0.75 use-bias=false time-offsets=0 orthonormal-constraint=-1.0"+"\n")
                    else:
                        out_f.write("component name="+tdnnf_name+".linear type=TdnnComponent input-dim=1536 output-dim=240 l2-regularize=0.01 max-change=0.75 use-bias=false time-offsets="+str(context_offset[(count-2)*2])+",0 orthonormal-constraint=-1.0"+"\n")
                    if count == 2:
                        out_f.write("component-node name="+tdnnf_name+".linear component="+tdnnf_name+".linear input=tdnn1.dropout"+"\n")
                    else:
                        out_f.write("component-node name="+tdnnf_name+".linear component="+tdnnf_name+".linear input="+tdnnfpre_name+".noop"+"\n")                  

                    out_f.write("dim-range-node name="+tdnnf_name+"0.linear input-node="+tdnnf_name+".linear dim-offset=0 dim=25"+"\n")
                    out_f.write("dim-range-node name="+tdnnf_name+"1.linear input-node="+tdnnf_name+".linear dim-offset=25 dim=25"+"\n")
                    out_f.write("dim-range-node name="+tdnnf_name+"2.linear input-node="+tdnnf_name+".linear dim-offset=50 dim=30"+"\n")
                    out_f.write("dim-range-node name="+tdnnf_name+"3.linear input-node="+tdnnf_name+".linear dim-offset=80 dim=20"+"\n") 
                    out_f.write("dim-range-node name="+tdnnf_name+"4.linear input-node="+tdnnf_name+".linear dim-offset=100 dim=20"+"\n")
                    out_f.write("dim-range-node name="+tdnnf_name+"5.linear input-node="+tdnnf_name+".linear dim-offset=120 dim=40"+"\n")
                    out_f.write("dim-range-node name="+tdnnf_name+"6.linear input-node="+tdnnf_name+".linear dim-offset=160 dim=40"+"\n")
                    out_f.write("dim-range-node name="+tdnnf_name+"7.linear input-node="+tdnnf_name+".linear dim-offset=200 dim=40"+"\n")

                    out_f.write("component name="+tdnnf_name+"0.output type=ElementwiseProductComponent input-dim=50 output-dim=25"+"\n")
                    out_f.write("component-node name="+tdnnf_name+"0.output component="+tdnnf_name+"0.output input=Append("+tdnnf_name+"0.copyn, "+tdnnf_name+"0.linear)"+"\n")
                    out_f.write("component name="+tdnnf_name+"1.output type=ElementwiseProductComponent input-dim=50 output-dim=25"+"\n")
                    out_f.write("component-node name="+tdnnf_name+"1.output component="+tdnnf_name+"1.output input=Append("+tdnnf_name+"1.copyn, "+tdnnf_name+"1.linear)"+"\n")
                    out_f.write("component name="+tdnnf_name+"2.output type=ElementwiseProductComponent input-dim=60 output-dim=30"+"\n")
                    out_f.write("component-node name="+tdnnf_name+"2.output component="+tdnnf_name+"2.output input=Append("+tdnnf_name+"2.copyn, "+tdnnf_name+"2.linear)"+"\n")
                    out_f.write("component name="+tdnnf_name+"3.output type=ElementwiseProductComponent input-dim=40 output-dim=20"+"\n")
                    out_f.write("component-node name="+tdnnf_name+"3.output component="+tdnnf_name+"3.output input=Append("+tdnnf_name+"3.copyn, "+tdnnf_name+"3.linear)"+"\n") 
                    out_f.write("component name="+tdnnf_name+"4.output type=ElementwiseProductComponent input-dim=40 output-dim=20"+"\n")
                    out_f.write("component-node name="+tdnnf_name+"4.output component="+tdnnf_name+"4.output input=Append("+tdnnf_name+"4.copyn, "+tdnnf_name+"4.linear)"+"\n") 
                    out_f.write("component name="+tdnnf_name+"5.output type=ElementwiseProductComponent input-dim=80 output-dim=40"+"\n")
                    out_f.write("component-node name="+tdnnf_name+"5.output component="+tdnnf_name+"5.output input=Append("+tdnnf_name+"5.copyn, "+tdnnf_name+"5.linear)"+"\n") 
                    out_f.write("component name="+tdnnf_name+"6.output type=ElementwiseProductComponent input-dim=80 output-dim=40"+"\n")
                    out_f.write("component-node name="+tdnnf_name+"6.output component="+tdnnf_name+"6.output input=Append("+tdnnf_name+"6.copyn, "+tdnnf_name+"6.linear)"+"\n") 
                    out_f.write("component name="+tdnnf_name+"7.output type=ElementwiseProductComponent input-dim=80 output-dim=40"+"\n")
                    out_f.write("component-node name="+tdnnf_name+"7.output component="+tdnnf_name+"7.output input=Append("+tdnnf_name+"7.copyn, "+tdnnf_name+"7.linear)"+"\n") 
                    if context_offset[(count-2)*2+1] == 0:
                        out_f.write("component name="+tdnnf_name+".affine type=TdnnComponent input-dim=240 output-dim=1536 l2-regularize=0.01 max-change=0.75 time-offsets=0"+"\n")
                    else:
                        out_f.write("component name="+tdnnf_name+".affine type=TdnnComponent input-dim=240 output-dim=1536 l2-regularize=0.01 max-change=0.75 time-offsets=0,"+str(context_offset[(count-2)*2+1])+"\n")
                    out_f.write("component-node name="+tdnnf_name+".affine component="+tdnnf_name+".affine input=Append("+tdnnf_name+"0.output,"+tdnnf_name+"1.output,"+tdnnf_name+"2.output,"+tdnnf_name+"3.output,"+tdnnf_name+"4.output,"+tdnnf_name+"5.output,"+tdnnf_name+"6.output,"+tdnnf_name+"7.output)"+"\n")
                        
                    out_f.write("component name="+tdnnf_name+".relu type=RectifiedLinearComponent dim=1536 self-repair-scale=1e-05"+"\n")
                    out_f.write("component-node name="+tdnnf_name+".relu component="+tdnnf_name+".relu input="+tdnnf_name+".affine"+"\n")
                    out_f.write("component name="+tdnnf_name+".batchnorm type=BatchNormComponent dim=1536"+"\n")
                    out_f.write("component-node name="+tdnnf_name+".batchnorm component="+tdnnf_name+".batchnorm input="+tdnnf_name+".relu"+"\n")                
                elif 'component-node name=tdnnf'+str(count)+'.linear' in line or 'component name=tdnnf'+str(count)+'.affine' in line or 'component-node name=tdnnf'+str(count)+'.affine' in line:
                    continue
                elif 'component-node name=tdnnf'+str(count)+'.relu' in line or 'component name=tdnnf'+str(count)+'.relu' in line:
                    continue
                elif 'component-node name=tdnnf'+str(count)+'.batchnorm' in line or 'component name=tdnnf'+str(count)+'.batchnorm' in line:
                    continue    
                elif 'component name=tdnnf'+str(count)+'.dropout' in line:
                    continue              
                elif 'component-node name=tdnnf'+str(count)+'.dropout' in line:
                    out_f.write("component name="+tdnnf_name+".dropout type=GeneralDropoutComponent dim=1536 dropout-proportion=0.0 continuous=true"+"\n")
                    out_f.write("component-node name="+tdnnf_name+".dropout component="+tdnnf_name+".dropout input="+tdnnf_name+".batchnorm"+"\n")  
                    count += 1
                else:
                    out_f.write(line+'\n')
elif network_type == 'cnn-tdnn':
    count = 7
    with open(config_path+'/final.config', 'w') as out_f:
        with open(config_path+'/final_ori.config') as f:
            for line in f.readlines():
                line = line.strip()
                if 'component name=tdnnf'+str(count)+'.linear' in line:
                    tdnnf_name = 'tdnnf'+str(count)
                    tdnnfpre_name = 'tdnnf'+str(count-1)
                    out_f.write("component name="+tdnnf_name+".softmax type=OnehotFunctionComponent input-dim=40 output-dim=8 is-updatable=true use-natural-gradient=false"+"\n")
                    out_f.write("component-node name="+tdnnf_name+".softmax component="+tdnnf_name+".softmax input=input"+"\n")
                    out_f.write("dim-range-node name="+tdnnf_name+".softmax0 input-node="+tdnnf_name+".softmax dim-offset=0 dim=1"+"\n")
                    out_f.write("dim-range-node name="+tdnnf_name+".softmax1 input-node="+tdnnf_name+".softmax dim-offset=1 dim=1"+"\n")
                    out_f.write("dim-range-node name="+tdnnf_name+".softmax2 input-node="+tdnnf_name+".softmax dim-offset=2 dim=1"+"\n")
                    out_f.write("dim-range-node name="+tdnnf_name+".softmax3 input-node="+tdnnf_name+".softmax dim-offset=3 dim=1"+"\n")
                    out_f.write("dim-range-node name="+tdnnf_name+".softmax4 input-node="+tdnnf_name+".softmax dim-offset=4 dim=1"+"\n")
                    out_f.write("dim-range-node name="+tdnnf_name+".softmax5 input-node="+tdnnf_name+".softmax dim-offset=5 dim=1"+"\n")
                    out_f.write("dim-range-node name="+tdnnf_name+".softmax6 input-node="+tdnnf_name+".softmax dim-offset=6 dim=1"+"\n")
                    out_f.write("dim-range-node name="+tdnnf_name+".softmax7 input-node="+tdnnf_name+".softmax dim-offset=7 dim=1"+"\n")

                    out_f.write("component name="+tdnnf_name+"0.copyn type=CopyNComponent input-dim=1 output-dim=25"+"\n")
                    out_f.write("component-node name="+tdnnf_name+"0.copyn component="+tdnnf_name+"0.copyn input=Sum("+tdnnf_name+".softmax0,"+tdnnf_name+".softmax1,"+tdnnf_name+".softmax2,"+tdnnf_name+".softmax3,"+tdnnf_name+".softmax4,"+tdnnf_name+".softmax5,"+tdnnf_name+".softmax6,"+tdnnf_name+".softmax7)"+"\n")
                    out_f.write("component name="+tdnnf_name+"1.copyn type=CopyNComponent input-dim=1 output-dim=25"+"\n")
                    out_f.write("component-node name="+tdnnf_name+"1.copyn component="+tdnnf_name+"1.copyn input=Sum("+tdnnf_name+".softmax1,"+tdnnf_name+".softmax2,"+tdnnf_name+".softmax3,"+tdnnf_name+".softmax4,"+tdnnf_name+".softmax5,"+tdnnf_name+".softmax6,"+tdnnf_name+".softmax7)"+"\n")
                    out_f.write("component name="+tdnnf_name+"2.copyn type=CopyNComponent input-dim=1 output-dim=30"+"\n")
                    out_f.write("component-node name="+tdnnf_name+"2.copyn component="+tdnnf_name+"2.copyn input=Sum("+tdnnf_name+".softmax2,"+tdnnf_name+".softmax3,"+tdnnf_name+".softmax4,"+tdnnf_name+".softmax5,"+tdnnf_name+".softmax6,"+tdnnf_name+".softmax7)"+"\n")
                    out_f.write("component name="+tdnnf_name+"3.copyn type=CopyNComponent input-dim=1 output-dim=20"+"\n")
                    out_f.write("component-node name="+tdnnf_name+"3.copyn component="+tdnnf_name+"3.copyn input=Sum("+tdnnf_name+".softmax3,"+tdnnf_name+".softmax4,"+tdnnf_name+".softmax5,"+tdnnf_name+".softmax6,"+tdnnf_name+".softmax7)"+"\n")
                    out_f.write("component name="+tdnnf_name+"4.copyn type=CopyNComponent input-dim=1 output-dim=20"+"\n")
                    out_f.write("component-node name="+tdnnf_name+"4.copyn component="+tdnnf_name+"4.copyn input=Sum("+tdnnf_name+".softmax4,"+tdnnf_name+".softmax5,"+tdnnf_name+".softmax6,"+tdnnf_name+".softmax7)"+"\n")
                    out_f.write("component name="+tdnnf_name+"5.copyn type=CopyNComponent input-dim=1 output-dim=40"+"\n")
                    out_f.write("component-node name="+tdnnf_name+"5.copyn component="+tdnnf_name+"5.copyn input=Sum("+tdnnf_name+".softmax5,"+tdnnf_name+".softmax6,"+tdnnf_name+".softmax7)"+"\n")
                    out_f.write("component name="+tdnnf_name+"6.copyn type=CopyNComponent input-dim=1 output-dim=40"+"\n")
                    out_f.write("component-node name="+tdnnf_name+"6.copyn component="+tdnnf_name+"6.copyn input=Sum("+tdnnf_name+".softmax6,"+tdnnf_name+".softmax7)"+"\n")
                    out_f.write("component name="+tdnnf_name+"7.copyn type=CopyNComponent input-dim=1 output-dim=40"+"\n")
                    out_f.write("component-node name="+tdnnf_name+"7.copyn component="+tdnnf_name+"7.copyn input="+tdnnf_name+".softmax7"+"\n")

                    if context_offset[(count-7)*2] == 0:
                        if count == 7:
                            out_f.write("component name="+tdnnf_name+".linear type=TdnnComponent input-dim=2560 output-dim=240 l2-regularize=0.01 max-change=0.75 use-bias=false time-offsets=0 orthonormal-constraint=-1.0"+"\n")
                        else:
                            out_f.write("component name="+tdnnf_name+".linear type=TdnnComponent input-dim=1536 output-dim=240 l2-regularize=0.01 max-change=0.75 use-bias=false time-offsets=0 orthonormal-constraint=-1.0"+"\n")
                    else:
                        if count == 7:
                            out_f.write("component name="+tdnnf_name+".linear type=TdnnComponent input-dim=2560 output-dim=240 l2-regularize=0.01 max-change=0.75 use-bias=false time-offsets="+str(context_offset[(count-7)*2])+",0 orthonormal-constraint=-1.0"+"\n")
                        else:
                            out_f.write("component name="+tdnnf_name+".linear type=TdnnComponent input-dim=1536 output-dim=240 l2-regularize=0.01 max-change=0.75 use-bias=false time-offsets="+str(context_offset[(count-7)*2])+",0 orthonormal-constraint=-1.0"+"\n")
                    if count == 7:
                        out_f.write("component-node name="+tdnnf_name+".linear component="+tdnnf_name+".linear input=cnn6.batchnorm"+"\n")
                    elif count == 8:
                        out_f.write("component-node name="+tdnnf_name+".linear component="+tdnnf_name+".linear input=tdnnf7.dropout"+"\n")
                    else:
                        out_f.write("component-node name="+tdnnf_name+".linear component="+tdnnf_name+".linear input="+tdnnfpre_name+".noop"+"\n")                  

                    out_f.write("dim-range-node name="+tdnnf_name+"0.linear input-node="+tdnnf_name+".linear dim-offset=0 dim=25"+"\n")
                    out_f.write("dim-range-node name="+tdnnf_name+"1.linear input-node="+tdnnf_name+".linear dim-offset=25 dim=25"+"\n")
                    out_f.write("dim-range-node name="+tdnnf_name+"2.linear input-node="+tdnnf_name+".linear dim-offset=50 dim=30"+"\n")
                    out_f.write("dim-range-node name="+tdnnf_name+"3.linear input-node="+tdnnf_name+".linear dim-offset=80 dim=20"+"\n") 
                    out_f.write("dim-range-node name="+tdnnf_name+"4.linear input-node="+tdnnf_name+".linear dim-offset=100 dim=20"+"\n")
                    out_f.write("dim-range-node name="+tdnnf_name+"5.linear input-node="+tdnnf_name+".linear dim-offset=120 dim=40"+"\n")
                    out_f.write("dim-range-node name="+tdnnf_name+"6.linear input-node="+tdnnf_name+".linear dim-offset=160 dim=40"+"\n")
                    out_f.write("dim-range-node name="+tdnnf_name+"7.linear input-node="+tdnnf_name+".linear dim-offset=200 dim=40"+"\n")

                    out_f.write("component name="+tdnnf_name+"0.output type=ElementwiseProductComponent input-dim=50 output-dim=25"+"\n")
                    out_f.write("component-node name="+tdnnf_name+"0.output component="+tdnnf_name+"0.output input=Append("+tdnnf_name+"0.copyn, "+tdnnf_name+"0.linear)"+"\n")
                    out_f.write("component name="+tdnnf_name+"1.output type=ElementwiseProductComponent input-dim=50 output-dim=25"+"\n")
                    out_f.write("component-node name="+tdnnf_name+"1.output component="+tdnnf_name+"1.output input=Append("+tdnnf_name+"1.copyn, "+tdnnf_name+"1.linear)"+"\n")
                    out_f.write("component name="+tdnnf_name+"2.output type=ElementwiseProductComponent input-dim=60 output-dim=30"+"\n")
                    out_f.write("component-node name="+tdnnf_name+"2.output component="+tdnnf_name+"2.output input=Append("+tdnnf_name+"2.copyn, "+tdnnf_name+"2.linear)"+"\n")
                    out_f.write("component name="+tdnnf_name+"3.output type=ElementwiseProductComponent input-dim=40 output-dim=20"+"\n")
                    out_f.write("component-node name="+tdnnf_name+"3.output component="+tdnnf_name+"3.output input=Append("+tdnnf_name+"3.copyn, "+tdnnf_name+"3.linear)"+"\n") 
                    out_f.write("component name="+tdnnf_name+"4.output type=ElementwiseProductComponent input-dim=40 output-dim=20"+"\n")
                    out_f.write("component-node name="+tdnnf_name+"4.output component="+tdnnf_name+"4.output input=Append("+tdnnf_name+"4.copyn, "+tdnnf_name+"4.linear)"+"\n") 
                    out_f.write("component name="+tdnnf_name+"5.output type=ElementwiseProductComponent input-dim=80 output-dim=40"+"\n")
                    out_f.write("component-node name="+tdnnf_name+"5.output component="+tdnnf_name+"5.output input=Append("+tdnnf_name+"5.copyn, "+tdnnf_name+"5.linear)"+"\n") 
                    out_f.write("component name="+tdnnf_name+"6.output type=ElementwiseProductComponent input-dim=80 output-dim=40"+"\n")
                    out_f.write("component-node name="+tdnnf_name+"6.output component="+tdnnf_name+"6.output input=Append("+tdnnf_name+"6.copyn, "+tdnnf_name+"6.linear)"+"\n") 
                    out_f.write("component name="+tdnnf_name+"7.output type=ElementwiseProductComponent input-dim=80 output-dim=40"+"\n")
                    out_f.write("component-node name="+tdnnf_name+"7.output component="+tdnnf_name+"7.output input=Append("+tdnnf_name+"7.copyn, "+tdnnf_name+"7.linear)"+"\n") 
                    if context_offset[(count-7)*2+1] == 0:
                        out_f.write("component name="+tdnnf_name+".affine type=TdnnComponent input-dim=240 output-dim=1536 l2-regularize=0.01 max-change=0.75 time-offsets=0"+"\n")
                    else:
                        out_f.write("component name="+tdnnf_name+".affine type=TdnnComponent input-dim=240 output-dim=1536 l2-regularize=0.01 max-change=0.75 time-offsets=0,"+str(context_offset[(count-7)*2+1])+"\n")
                    out_f.write("component-node name="+tdnnf_name+".affine component="+tdnnf_name+".affine input=Append("+tdnnf_name+"0.output,"+tdnnf_name+"1.output,"+tdnnf_name+"2.output,"+tdnnf_name+"3.output,"+tdnnf_name+"4.output,"+tdnnf_name+"5.output,"+tdnnf_name+"6.output,"+tdnnf_name+"7.output)"+"\n")
                        
                    out_f.write("component name="+tdnnf_name+".relu type=RectifiedLinearComponent dim=1536 self-repair-scale=1e-05"+"\n")
                    out_f.write("component-node name="+tdnnf_name+".relu component="+tdnnf_name+".relu input="+tdnnf_name+".affine"+"\n")
                    out_f.write("component name="+tdnnf_name+".batchnorm type=BatchNormComponent dim=1536"+"\n")
                    out_f.write("component-node name="+tdnnf_name+".batchnorm component="+tdnnf_name+".batchnorm input="+tdnnf_name+".relu"+"\n")                
                elif 'component-node name=tdnnf'+str(count)+'.linear' in line or 'component name=tdnnf'+str(count)+'.affine' in line or 'component-node name=tdnnf'+str(count)+'.affine' in line:
                    continue
                elif 'component-node name=tdnnf'+str(count)+'.relu' in line or 'component name=tdnnf'+str(count)+'.relu' in line:
                    continue
                elif 'component-node name=tdnnf'+str(count)+'.batchnorm' in line or 'component name=tdnnf'+str(count)+'.batchnorm' in line:
                    continue    
                elif 'component name=tdnnf'+str(count)+'.dropout' in line:
                    continue              
                elif 'component-node name=tdnnf'+str(count)+'.dropout' in line:
                    out_f.write("component name="+tdnnf_name+".dropout type=GeneralDropoutComponent dim=1536 dropout-proportion=0.0 continuous=true"+"\n")
                    out_f.write("component-node name="+tdnnf_name+".dropout component="+tdnnf_name+".dropout input="+tdnnf_name+".batchnorm"+"\n")  
                    count += 1
                else:
                    out_f.write(line+'\n')


    count = 7
    with open(config_path+'/ref.config', 'w') as out_f:
        with open(config_path+'/ref_ori.config') as f:
            for line in f.readlines():
                line = line.strip()
                if 'component name=tdnnf'+str(count)+'.linear' in line:
                    tdnnf_name = 'tdnnf'+str(count)
                    tdnnfpre_name = 'tdnnf'+str(count-1)
                    out_f.write("component name="+tdnnf_name+".softmax type=OnehotFunctionComponent input-dim=40 output-dim=8 is-updatable=true use-natural-gradient=false"+"\n")
                    out_f.write("component-node name="+tdnnf_name+".softmax component="+tdnnf_name+".softmax input=input"+"\n")
                    out_f.write("dim-range-node name="+tdnnf_name+".softmax0 input-node="+tdnnf_name+".softmax dim-offset=0 dim=1"+"\n")
                    out_f.write("dim-range-node name="+tdnnf_name+".softmax1 input-node="+tdnnf_name+".softmax dim-offset=1 dim=1"+"\n")
                    out_f.write("dim-range-node name="+tdnnf_name+".softmax2 input-node="+tdnnf_name+".softmax dim-offset=2 dim=1"+"\n")
                    out_f.write("dim-range-node name="+tdnnf_name+".softmax3 input-node="+tdnnf_name+".softmax dim-offset=3 dim=1"+"\n")
                    out_f.write("dim-range-node name="+tdnnf_name+".softmax4 input-node="+tdnnf_name+".softmax dim-offset=4 dim=1"+"\n")
                    out_f.write("dim-range-node name="+tdnnf_name+".softmax5 input-node="+tdnnf_name+".softmax dim-offset=5 dim=1"+"\n")
                    out_f.write("dim-range-node name="+tdnnf_name+".softmax6 input-node="+tdnnf_name+".softmax dim-offset=6 dim=1"+"\n")
                    out_f.write("dim-range-node name="+tdnnf_name+".softmax7 input-node="+tdnnf_name+".softmax dim-offset=7 dim=1"+"\n")

                    out_f.write("component name="+tdnnf_name+"0.copyn type=CopyNComponent input-dim=1 output-dim=25"+"\n")
                    out_f.write("component-node name="+tdnnf_name+"0.copyn component="+tdnnf_name+"0.copyn input=Sum("+tdnnf_name+".softmax0,"+tdnnf_name+".softmax1,"+tdnnf_name+".softmax2,"+tdnnf_name+".softmax3,"+tdnnf_name+".softmax4,"+tdnnf_name+".softmax5,"+tdnnf_name+".softmax6,"+tdnnf_name+".softmax7)"+"\n")
                    out_f.write("component name="+tdnnf_name+"1.copyn type=CopyNComponent input-dim=1 output-dim=25"+"\n")
                    out_f.write("component-node name="+tdnnf_name+"1.copyn component="+tdnnf_name+"1.copyn input=Sum("+tdnnf_name+".softmax1,"+tdnnf_name+".softmax2,"+tdnnf_name+".softmax3,"+tdnnf_name+".softmax4,"+tdnnf_name+".softmax5,"+tdnnf_name+".softmax6,"+tdnnf_name+".softmax7)"+"\n")
                    out_f.write("component name="+tdnnf_name+"2.copyn type=CopyNComponent input-dim=1 output-dim=30"+"\n")
                    out_f.write("component-node name="+tdnnf_name+"2.copyn component="+tdnnf_name+"2.copyn input=Sum("+tdnnf_name+".softmax2,"+tdnnf_name+".softmax3,"+tdnnf_name+".softmax4,"+tdnnf_name+".softmax5,"+tdnnf_name+".softmax6,"+tdnnf_name+".softmax7)"+"\n")
                    out_f.write("component name="+tdnnf_name+"3.copyn type=CopyNComponent input-dim=1 output-dim=20"+"\n")
                    out_f.write("component-node name="+tdnnf_name+"3.copyn component="+tdnnf_name+"3.copyn input=Sum("+tdnnf_name+".softmax3,"+tdnnf_name+".softmax4,"+tdnnf_name+".softmax5,"+tdnnf_name+".softmax6,"+tdnnf_name+".softmax7)"+"\n")
                    out_f.write("component name="+tdnnf_name+"4.copyn type=CopyNComponent input-dim=1 output-dim=20"+"\n")
                    out_f.write("component-node name="+tdnnf_name+"4.copyn component="+tdnnf_name+"4.copyn input=Sum("+tdnnf_name+".softmax4,"+tdnnf_name+".softmax5,"+tdnnf_name+".softmax6,"+tdnnf_name+".softmax7)"+"\n")
                    out_f.write("component name="+tdnnf_name+"5.copyn type=CopyNComponent input-dim=1 output-dim=40"+"\n")
                    out_f.write("component-node name="+tdnnf_name+"5.copyn component="+tdnnf_name+"5.copyn input=Sum("+tdnnf_name+".softmax5,"+tdnnf_name+".softmax6,"+tdnnf_name+".softmax7)"+"\n")
                    out_f.write("component name="+tdnnf_name+"6.copyn type=CopyNComponent input-dim=1 output-dim=40"+"\n")
                    out_f.write("component-node name="+tdnnf_name+"6.copyn component="+tdnnf_name+"6.copyn input=Sum("+tdnnf_name+".softmax6,"+tdnnf_name+".softmax7)"+"\n")
                    out_f.write("component name="+tdnnf_name+"7.copyn type=CopyNComponent input-dim=1 output-dim=40"+"\n")
                    out_f.write("component-node name="+tdnnf_name+"7.copyn component="+tdnnf_name+"7.copyn input="+tdnnf_name+".softmax7"+"\n")

                    if context_offset[(count-7)*2] == 0:
                        if count == 7:
                            out_f.write("component name="+tdnnf_name+".linear type=TdnnComponent input-dim=2560 output-dim=240 l2-regularize=0.01 max-change=0.75 use-bias=false time-offsets=0 orthonormal-constraint=-1.0"+"\n")
                        else:
                            out_f.write("component name="+tdnnf_name+".linear type=TdnnComponent input-dim=1536 output-dim=240 l2-regularize=0.01 max-change=0.75 use-bias=false time-offsets=0 orthonormal-constraint=-1.0"+"\n")
                    else:
                        if count == 7:
                            out_f.write("component name="+tdnnf_name+".linear type=TdnnComponent input-dim=2560 output-dim=240 l2-regularize=0.01 max-change=0.75 use-bias=false time-offsets="+str(context_offset[(count-7)*2])+",0 orthonormal-constraint=-1.0"+"\n")
                        else:
                            out_f.write("component name="+tdnnf_name+".linear type=TdnnComponent input-dim=1536 output-dim=240 l2-regularize=0.01 max-change=0.75 use-bias=false time-offsets="+str(context_offset[(count-7)*2])+",0 orthonormal-constraint=-1.0"+"\n")
                    if count == 7:
                        out_f.write("component-node name="+tdnnf_name+".linear component="+tdnnf_name+".linear input=cnn6.batchnorm"+"\n")
                    elif count == 8:
                        out_f.write("component-node name="+tdnnf_name+".linear component="+tdnnf_name+".linear input=tdnnf7.dropout"+"\n")
                    else:
                        out_f.write("component-node name="+tdnnf_name+".linear component="+tdnnf_name+".linear input="+tdnnfpre_name+".noop"+"\n")                  
                    out_f.write("dim-range-node name="+tdnnf_name+"0.linear input-node="+tdnnf_name+".linear dim-offset=0 dim=25"+"\n")
                    out_f.write("dim-range-node name="+tdnnf_name+"1.linear input-node="+tdnnf_name+".linear dim-offset=25 dim=25"+"\n")
                    out_f.write("dim-range-node name="+tdnnf_name+"2.linear input-node="+tdnnf_name+".linear dim-offset=50 dim=30"+"\n")
                    out_f.write("dim-range-node name="+tdnnf_name+"3.linear input-node="+tdnnf_name+".linear dim-offset=80 dim=20"+"\n") 
                    out_f.write("dim-range-node name="+tdnnf_name+"4.linear input-node="+tdnnf_name+".linear dim-offset=100 dim=20"+"\n")
                    out_f.write("dim-range-node name="+tdnnf_name+"5.linear input-node="+tdnnf_name+".linear dim-offset=120 dim=40"+"\n")
                    out_f.write("dim-range-node name="+tdnnf_name+"6.linear input-node="+tdnnf_name+".linear dim-offset=160 dim=40"+"\n")
                    out_f.write("dim-range-node name="+tdnnf_name+"7.linear input-node="+tdnnf_name+".linear dim-offset=200 dim=40"+"\n")

                    out_f.write("component name="+tdnnf_name+"0.output type=ElementwiseProductComponent input-dim=50 output-dim=25"+"\n")
                    out_f.write("component-node name="+tdnnf_name+"0.output component="+tdnnf_name+"0.output input=Append("+tdnnf_name+"0.copyn, "+tdnnf_name+"0.linear)"+"\n")
                    out_f.write("component name="+tdnnf_name+"1.output type=ElementwiseProductComponent input-dim=50 output-dim=25"+"\n")
                    out_f.write("component-node name="+tdnnf_name+"1.output component="+tdnnf_name+"1.output input=Append("+tdnnf_name+"1.copyn, "+tdnnf_name+"1.linear)"+"\n")
                    out_f.write("component name="+tdnnf_name+"2.output type=ElementwiseProductComponent input-dim=60 output-dim=30"+"\n")
                    out_f.write("component-node name="+tdnnf_name+"2.output component="+tdnnf_name+"2.output input=Append("+tdnnf_name+"2.copyn, "+tdnnf_name+"2.linear)"+"\n")
                    out_f.write("component name="+tdnnf_name+"3.output type=ElementwiseProductComponent input-dim=40 output-dim=20"+"\n")
                    out_f.write("component-node name="+tdnnf_name+"3.output component="+tdnnf_name+"3.output input=Append("+tdnnf_name+"3.copyn, "+tdnnf_name+"3.linear)"+"\n") 
                    out_f.write("component name="+tdnnf_name+"4.output type=ElementwiseProductComponent input-dim=40 output-dim=20"+"\n")
                    out_f.write("component-node name="+tdnnf_name+"4.output component="+tdnnf_name+"4.output input=Append("+tdnnf_name+"4.copyn, "+tdnnf_name+"4.linear)"+"\n") 
                    out_f.write("component name="+tdnnf_name+"5.output type=ElementwiseProductComponent input-dim=80 output-dim=40"+"\n")
                    out_f.write("component-node name="+tdnnf_name+"5.output component="+tdnnf_name+"5.output input=Append("+tdnnf_name+"5.copyn, "+tdnnf_name+"5.linear)"+"\n") 
                    out_f.write("component name="+tdnnf_name+"6.output type=ElementwiseProductComponent input-dim=80 output-dim=40"+"\n")
                    out_f.write("component-node name="+tdnnf_name+"6.output component="+tdnnf_name+"6.output input=Append("+tdnnf_name+"6.copyn, "+tdnnf_name+"6.linear)"+"\n") 
                    out_f.write("component name="+tdnnf_name+"7.output type=ElementwiseProductComponent input-dim=80 output-dim=40"+"\n")
                    out_f.write("component-node name="+tdnnf_name+"7.output component="+tdnnf_name+"7.output input=Append("+tdnnf_name+"7.copyn, "+tdnnf_name+"7.linear)"+"\n") 
                    if context_offset[(count-7)*2+1] == 0:
                        out_f.write("component name="+tdnnf_name+".affine type=TdnnComponent input-dim=240 output-dim=1536 l2-regularize=0.01 max-change=0.75 time-offsets=0"+"\n")
                    else:
                        out_f.write("component name="+tdnnf_name+".affine type=TdnnComponent input-dim=240 output-dim=1536 l2-regularize=0.01 max-change=0.75 time-offsets=0,"+str(context_offset[(count-7)*2+1])+"\n")
                    out_f.write("component-node name="+tdnnf_name+".affine component="+tdnnf_name+".affine input=Append("+tdnnf_name+"0.output,"+tdnnf_name+"1.output,"+tdnnf_name+"2.output,"+tdnnf_name+"3.output,"+tdnnf_name+"4.output,"+tdnnf_name+"5.output,"+tdnnf_name+"6.output,"+tdnnf_name+"7.output)"+"\n")
                        
                    out_f.write("component name="+tdnnf_name+".relu type=RectifiedLinearComponent dim=1536 self-repair-scale=1e-05"+"\n")
                    out_f.write("component-node name="+tdnnf_name+".relu component="+tdnnf_name+".relu input="+tdnnf_name+".affine"+"\n")
                    out_f.write("component name="+tdnnf_name+".batchnorm type=BatchNormComponent dim=1536"+"\n")
                    out_f.write("component-node name="+tdnnf_name+".batchnorm component="+tdnnf_name+".batchnorm input="+tdnnf_name+".relu"+"\n")                
                elif 'component-node name=tdnnf'+str(count)+'.linear' in line or 'component name=tdnnf'+str(count)+'.affine' in line or 'component-node name=tdnnf'+str(count)+'.affine' in line:
                    continue
                elif 'component-node name=tdnnf'+str(count)+'.relu' in line or 'component name=tdnnf'+str(count)+'.relu' in line:
                    continue
                elif 'component-node name=tdnnf'+str(count)+'.batchnorm' in line or 'component name=tdnnf'+str(count)+'.batchnorm' in line:
                    continue    
                elif 'component name=tdnnf'+str(count)+'.dropout' in line:
                    continue              
                elif 'component-node name=tdnnf'+str(count)+'.dropout' in line:
                    out_f.write("component name="+tdnnf_name+".dropout type=GeneralDropoutComponent dim=1536 dropout-proportion=0.0 continuous=true"+"\n")
                    out_f.write("component-node name="+tdnnf_name+".dropout component="+tdnnf_name+".dropout input="+tdnnf_name+".batchnorm"+"\n")  
                    count += 1
                else:
                    out_f.write(line+'\n')
