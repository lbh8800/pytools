function main()
%%  
%该代码根据已生成的xml，制作VOC2007数据集中的trainval.txt;train.txt;test.txt和val.txt  
%trainval占总数据集的50%，test占总数据集的50%；train占trainval的50%，val占trainval的50%；  
%上面所占百分比可根据自己的数据集修改，如果数据集比较少，test和val可少一些  
%%  
%注意修改下面四个值  
xmlfilepath='I:\xiaobai\Annotations';  
txtsavepath='I:\xiaobai\ImageSets\Main\';  
trainval_percent=0.7;%trainval占整个数据集的百分比，剩下部分就是test所占百分比  
train_percent=0.7;%train占trainval的百分比，剩下部分就是val所占百分比  
  
  
%%  
xmlfile=dir(xmlfilepath);  
numOfxml=length(xmlfile)-2;%减去.和..  总的数据集大小  
  
  
trainval=sort(randperm(numOfxml,floor(numOfxml*trainval_percent)));  
test=sort(setdiff(1:numOfxml,trainval));  
  
  
trainvalsize=length(trainval);%trainval的大小  
train=sort(trainval(randperm(trainvalsize,floor(trainvalsize*train_percent))));  
val=sort(setdiff(trainval,train));  
  
  
ftrainval=fopen([txtsavepath 'trainval.txt'],'wt');  
ftest=fopen([txtsavepath 'test.txt'],'wt');  
ftrain=fopen([txtsavepath 'train.txt'],'wt');  
fval=fopen([txtsavepath 'val.txt'],'wt');  
  
  
for i=1:numOfxml  
    if ismember(i,trainval)  
        fprintf(ftrainval,'%s\n',xmlfile(i+2).name(1:end-4));  
        if ismember(i,train)  
            fprintf(ftrain,'%s\n',xmlfile(i+2).name(1:end-4));  
        else  
            fprintf(fval,'%s\n',xmlfile(i+2).name(1:end-4));  
        end  
    else  
        fprintf(ftest,'%s\n',xmlfile(i+2).name(1:end-4));  
    end  
end  
fclose(ftrainval);  
fclose(ftrain);  
fclose(fval);  
fclose(ftest);  
end