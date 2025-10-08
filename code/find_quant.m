function [q,p,val_opt,val,time]=find_quant(alpha,x_train,y_train,x_cal,scores,q_naive,upper_loss,sc_cal,Y,mosek)


n_cal=length(x_cal(1,:));
n_train=length(x_train(1,:));
d=length(x_train(:,1));
y_ind=zeros(n_cal,Y);


for i=1:n_cal
for j=1:Y
    y_ind(i,j)=double(scores(i,j)<=q_naive);
end

end

sigmas=(10.^linspace(-2,1,10))*sqrt(d/2);



[sigma,val] = select_sigma(x_train,y_train,x_cal,y_ind,sigmas,Y);





G=[x_train,x_cal]'*[x_train,x_cal];
g=diag(G);
M=-0.5*(g*ones(1,n_train+n_cal)+ones(n_train+n_cal,1)*g'-2*G);
k=exp(M/(sigma^2));



[p,val_opt,time]=find_p(k,x_cal,x_train,y_train,n_train,n_cal,upper_loss,sc_cal,Y,mosek);


 q=weighted_quantile(scores(:),p/n_cal,(n_cal+1)*(1-alpha)/n_cal);


 

