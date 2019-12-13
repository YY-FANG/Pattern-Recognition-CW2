load('face');
X = X ./ (ones(size(X,1),1) * sqrt(sum(X.*X)));
train_X = X(:, 1:320);
train_l = l(1:320);
test_X = X(:, 321:520);
test_l = l(321:520);

[means, covariances, priors] =vl_gmm(train_X, 3);

for i = 1:size(test_X,2)
    encoding(:,i) = vl_fisher(test_X(:,i), means, covariances, priors);
end
% d=d';
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
%u=[4;3];
%v=[1,-1;-1,2];
u=[0.0214;0.0209];%均值
v=[4,3;3,9];%协方差阵
x=-7:0.05:7;
y=-7:0.05:7;
%x=-2:0.05:10;
%y=-3:0.05:9;
[X,Y]=meshgrid(x,y);
s2x=v(1,1)%x的方差
s2y=v(2,2)
sx=sqrt(s2x)%标准差
sy=sqrt(s2y)
Cov=v(1,2)
r=Cov/(sx*sy)
a=1/(2*pi*sx*sy*sqrt(1-r^2));
b1=-1/(2*(1-r^2));
b2=((X-u(1))./sx).^2;
b3=((Y-u(2))./sy).^2;
b4=2*r.*(X-u(1)).*(Y-u(2))./(sx*sy)
Z=a*exp(b1*(b2+b3-b4));%也就是f(x1,x2)的表达式
%mesh(x,y,Z);
%figure
%grid on
mesh(X,Y,Z),title('密度函数图')
figure
contour(X,Y,Z),title('等高线图')

