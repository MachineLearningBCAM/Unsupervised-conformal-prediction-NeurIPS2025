function [sigma,val] = select_sigma(x_train,y_train,x_cal,y_ind,sigmas,Y)

n_train=length(x_train(1,:));
n_cal=length(x_cal(1,:));

 H=length(sigmas);



Gtrain=x_train'*x_train;
gtrain=diag(Gtrain);
Mtrain=-0.5*(gtrain*ones(1,n_train)+ones(n_train,1)*gtrain'-2*Gtrain);

G=x_cal'*x_cal;
g=diag(G);
M=-0.5*(g*ones(1,n_cal)+ones(n_cal,1)*g'-2*G);
 opts.SYM=true;
 opts.POSDEF=true;

 y_ind_col=[];
for j=1:Y
    y_ind_col=[y_ind_col;y_ind(:,j)];
end

y_ind_col=sparse(y_ind_col);

KK=sparse(n_cal*Y,n_cal*Y);

for i=1:H

Ktrain=exp(Mtrain/(sigmas(i)^2));
K=double(exp(M/(sigmas(i)^2)));
K(abs(K) < 10^-8) = 0;
K=sparse(K);

for j=1:Y
KK((j-1)*n_cal+1:(j-1)*n_cal+n_cal,(j-1)*n_cal+1:(j-1)*n_cal+n_cal)=K+(10^(-10))*speye(n_cal);
end


alpha=mldivide(KK,y_ind_col);


val(i)=sqrt(alpha'*KK*alpha);


end

[~,idx]=min(val);
sigma=sigmas(idx);

