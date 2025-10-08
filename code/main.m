clear all
close all




mosek=0; %set to 0 if you dont want to use cvx mosek solver, it will use the matlab function instead

cd ..
cd data
load("usps.mat")
cd ..
cd code


Y=length(unique(y));
x=zscore(x')';

alpha=0.1;


d=length(x(:,1));

n_train=3000;
n_cal=1000;
n_test=1000;

idx=randperm(length(y));
x_train=x(:,idx(1:n_train));
y_train=y(idx(1:n_train));
x_test=x(:,idx(n_train+1:n_train+n_test));
y_test=y(idx(n_train+1:n_train+n_test));

modelCV=fitcnet(x_train',y_train,'LayerSizes',[128,64],'Lambda',0.001,'CrossVal','on');
model=fitcnet(x_train',y_train,'LayerSizes',[128,64],'Lambda',0.001);

[lab,sco]=kfoldPredict(modelCV);


loss=[];
for i=1:n_train
    loss(i)=-log(sco(i,y_train(i))+10^-40);
end

upper_loss=mean(loss)+std(loss)/sqrt(n_train);

score_train=zeros(n_train,1);

for i=1:n_train
    [~,probs]=predict(model,x_train(:,i)');

    score_train(i)=compute_score(probs,y_train(i));
  % score_train(i)=1-probs(y_train(i)); %compute_score function uses the
  % adaptive conformal score, use this line instead if you want to use the
  % score based on probability estimates

end




score_test=zeros(n_test,1);
scores_test=zeros(n_test,Y);

err=0;

for i=1:n_test
    [y_pred,probs]=predict(model,x_test(:,i)');

    score_test(i)=compute_score(probs,y_test(i));

    for j=1:Y
  %  scores_test(i,j)=1-probs(j);%compute_score function uses the
  % adaptive conformal score, use this line instead if you want to use the
  % score based on probability estimates
    scores_test(i,j)=compute_score(probs,j);

    end

    if y_pred~=y_test(i)
        err=err+1/n_test;
    end


end

x_cal=x(:,idx(n_train+n_test+1:n_train+n_test+n_cal));
y_cal=y(idx(n_train+n_test+1:n_train+n_test+n_cal));
x_tr=x_train(:,1:n_cal);
y_tr=y_train(1:n_cal);

score_cal=zeros(n_cal,1);
scores_cal=zeros(n_cal,Y);
preds_cal=zeros(n_cal,1);
sc_cal=zeros(n_cal,Y);
score_cal_pred=zeros(n_cal,1);


for i=1:n_cal
    [preds_cal(i),sc_cal(i,:)]=predict(model,x_cal(:,i)');

    score_cal(i)=compute_score(sc_cal(i,:),y_cal(i));
score_cal_pred(i)=compute_score(sc_cal(i,:),preds_cal(i));
%   score_cal(i)=1-sc_cal(i,y_cal(i));
 %   score_cal_pred(i)=1-sc_cal(i,preds_cal(i));%compute_score function uses the
  % adaptive conformal score, use this line instead if you want to use the
  % score based on probability estimates
   


    for j=1:Y
   % scores_cal(i,j)=1-sc_cal(i,j);%compute_score function uses the
  % adaptive conformal score, use this line instead if you want to use the
  % score based on probability estimates
   scores_cal(i,j)=compute_score(sc_cal(i,:),j);

    end

end




q_naive=weighted_quantile(score_cal_pred,(1/n_cal)*ones(1,n_cal),(1-alpha)*(1+1/n_cal));

q_sup=weighted_quantile(score_cal,(1/n_cal)*ones(1,n_cal),(1-alpha)*(1+1/n_cal));

q_uns=find_quant(alpha,x_tr,y_tr,x_cal,scores_cal,q_naive,upper_loss,sc_cal,Y,mosek);


cove_sup=zeros(n_test,1);
cove_uns=zeros(n_test,1);
cove_naive=zeros(n_test,1);
size_sup=zeros(n_test,1);
size_uns=zeros(n_test,1);
size_naive=zeros(n_test,1);

for i=1:n_test
idx_uns=find(scores_test(i,:)<=q_uns);
if sum(idx_uns==y_test(i))==1
    cove_uns(i)=1;
end
size_uns(i)=length(idx_uns);


idx_sup=find(scores_test(i,:)<=q_sup);
if sum(idx_sup==y_test(i))==1
    cove_sup(i)=1;
end
size_sup(i)=length(idx_sup);

idx_naive=find(scores_test(i,:)<=q_naive);
if sum(idx_naive==y_test(i))==1
    cove_naive(i)=1;
end
size_naive(i)=length(idx_naive);


end

ave_cove_uns=mean(cove_uns);
ave_size_uns=mean(size_uns);
ave_cove_sup=mean(cove_sup);
ave_size_sup=mean(size_sup);
ave_cove_naive=mean(cove_naive);
ave_size_naive=mean(size_naive);




       
