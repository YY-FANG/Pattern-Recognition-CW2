%% Data process
clear all
clc
load('face');

faces = reshape(X, [56,46,520]);
X = X ./ (ones(size(X,1),1) * sqrt(sum(X.*X)));
train_X = X(:, 1:320);
train_l = l(1:320);
test_X = X(:, 321:520);
test_l = l(321:520);
a=[];
P=[];
NP=[];
NUM=[];
clu=32;
label=[];
Lrest=[];
rank=1;
In=[];
InT=[];
[Idx,Ctrs,SumD,D] = kmeans(train_X',clu,'Replicates',100);
u=unique(Idx');
for i=1:clu
    A =find(Idx'==u(i));
    if isempty(a)
        a=[a;A];
    else
        l=max([size(a,2),length(A)]);
        A=[A , zeros(1,l-length(A))];
        a=[a , zeros(i-1,l-size(a,2))];
        a=[a;A];
    end
end %聚类可视化，形成矩阵，补零保证维度相同
for i=1:clu
    for j=1:size(a,2)
        if a(i,j)==0
            P(i,j)=0;
        else
            P(i,j)=train_l(a(i,j));
        end
    end
end  %替换为真实标签
Ctrs=Ctrs';
%%
% problem b)
% d = softmax_proba(test_X, Ctrs);
% d=d';
% AB = -2 * (d') * d; % (欧式距离和余弦相似度二选一）     
% BB = sum(d .* d);         
% AA = sum(d.* d);   
% distance = sqrt(bsxfun(@plus, bsxfun(@plus, AB, BB), AA'));
% % 
% [KNNfvB,KNNfvI]  = mink(distance,rank+1,2);%取前k+1项
% for p=1:200
%     Ir=KNNfvI(p,1:rank+1);
%      De = find(Ir == p); 
%      Ir(De)=[];
%      In=[In; test_l(Ir)];
% end
%% Recognition rate 
%  matchcount = 0;
% for k=1:200
%     
%         labelT = find(In(k,1:rank) == test_l(k)); 
%      
%         if ~isempty(labelT)
%             matchcount = matchcount + 1;
%         end
%     
% end
% rate=matchcount/200;
% for i=1:200
%     for j=1:32
%         
%         
%         d(i,j)=(test_X(:,i)-Ctrs(:,j))'*(test_X(:,i)-Ctrs(:,j));
%     end
% end


%% label allocation algorithm
% for i=1:32
%     L = unique(P(i,:));
%     NUM(i)=numel(L);
% end %提取每个类中的种类数量
% NUM1=NUM;
% for i=1:32
%     [OV,Or]=min(NUM1);
%     NUM1(Or)=NaN;
%     NP(i,:)=P(Or,:);
% end %将矩阵行按种类数量从低到高排序
% for i=1:32 %每行取max数量的种类作为标签
%     b=[];
%     Pi=NP(i,:);
%     B= Pi(Pi~=0);
%     c=mode(B);
%     for j=1:numel(label)
%         b=[b find(label(j)==c)];%检查是否发生类别重复
%     end
%     
%     if ~isempty(b) %重复进入循环持续迭代
%         while(1)
%             b=[];
%             B= B(B~=c);
%             if isempty(B)
%                 c=0;
%                 break; %B空值退出循环 赋值0
%             else
%                 c=mode(B);
%                 
%                 for j=1:numel(label)
%                     b=[b find(label(j)==c)];
%                 end
%                 if isempty(b) %无重复退出循环
%                     break;
%                 end
%             end
%         end
%     end
%     
%     label=[label; c]; %形成标签矩阵
%     
% end
% NV=find(label==0);%寻找0位置
% for i=1:numel(label)
%     
%     rest=find(label==i);
%     if isempty(rest)
%         Lrest=[Lrest i];
%     end
% end  %寻找未分配的值
% for i=1:numel(NV)
%     label(NV(i))=Lrest(i); %分配未分配的类给0位置
% end
% labe=sort(label);
%% perform @rank1
[min_a,index]=min(D,[],1);
% %% problem a)
d = codebook_distance_matrix(test_X, Ctrs, clu); %计算codebook distance matrix
% 11111111111111111111111111111111111111111KNN
d=d';
AB = -2 * (d') * d; % (欧式距离和余弦相似度二选一）     
BB = sum(d .* d);         
AA = sum(d.* d);   
distance = sqrt(bsxfun(@plus, bsxfun(@plus, AB, BB), AA'));
% 
[KNNB,KNNI]  = mink(distance,rank+1,2);%取前k+1项
for p=1:200
    Ir=KNNI(p,1:rank+1);
     De = find(Ir == p); 
     Ir(De)=[];
     In=[In; test_l(Ir)];
end
% Recognition rate 
 matchcount = 0;
for k=1:200
    
        labelT = find(In(k,1:rank) == test_l(k)); 
     
        if ~isempty(labelT)
            matchcount = matchcount + 1;
        end
    
end
rate=matchcount/200;

%% AP caculation
% i=199;
% [BT,IT]  = mink(distance,i+1,2);
% for p=1:200
%     IrT=IT(p,1:i+1);
%      DeT = find(IrT == p); 
%      IrT(DeT)=[];
%      InT=[InT; test_l(IrT)];
% end
% for p=1:200
%    SAP=0;
%     rankn = find(InT(p,1:199) == test_l(p)); 
%      
%     for h=1:numel(rankn)
%         Pr(p,h)=h/rankn(h);
%         RC(p,h)=h/numel(rankn);
%     end
% end
% for j=1:200
% for h=0:10
%     RCAP=h/10;
%     NewM = find(RC(j,:) >= RCAP);
%     pinterp(j,h+1) = max(Pr(j,NewM(:)));
% end
% end
% AP=mean(pinterp,2);
% mAP=mean(AP);


%% Accuracy
matchcount = 0;
for k=1:numel(label)
    
    Predict = find(NP(k,:) ==label(k) );
    
    if ~isempty(Predict)
        matchcount = matchcount + numel(Predict);
    end
    
end
rateT=matchcount/320;

% problem c)
u=ctrs
SL = (sta_face'*sta_face)/200;

% [means, covariances, priors] =vl_gmm(train_X, 3);
% 
% for i = 1:size(test_X,2)
%     encoding(:,i) = vl_fisher(test_X(:,i), means, covariances, priors);
% end
% % d=d';
% AB = -2 * (encoding') * encoding; % (欧式距离和余弦相似度二选一）     
% BB = sum(encoding .* encoding);         
% AA = sum(encoding.* encoding);   
% distance = sqrt(bsxfun(@plus, bsxfun(@plus, AB, BB), AA'));
% % 
% [KNNfvB,KNNfvI]  = mink(distance,rank+1,2);%取前k+1项
% for p=1:200
%     Ir=KNNfvI(p,1:rank+1);
%      De = find(Ir == p); 
%      Ir(De)=[];
%      In=[In; test_l(Ir)];
% end
% %% Recognition rate 
%  matchcount = 0;
% for k=1:200
%     
%         labelT = find(In(k,1:rank) == test_l(k)); 
%      
%         if ~isempty(labelT)
%             matchcount = matchcount + 1;
%         end
%     
% end
% rate=matchcount/200;
% [x,y]=meshgrid(linspace(-8,8,50));
% mu=[0 0];%期望
% sigma=[1 0;0 1];%协方差
% p=mvnpdf([x(:),y(:)],mu,sigma);%概率
% surf(x,y,reshape(p,size(x,1),[]))
