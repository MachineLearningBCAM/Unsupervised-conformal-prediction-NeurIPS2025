function  [p,val,time]=find_p(k,x_cal,x_train,y_train,n_train,n_cal,upper_loss,sc_cal,Y,mosek)


Kc=double(kron(eye(Y),k(n_train+1:n_train+n_cal,n_train+1:n_train+n_cal))+(10^-8)*eye(Y*n_cal));
v=zeros(n_cal*Y,1);
cont=1;
for i=1:Y
        idx=[];
        idx=find(y_train==i);
    for j=1:n_cal

        v(cont)=sum(k(idx,n_train+j));
        cont=cont+1;
    end
end

Ktval=0;
for i=1:n_train
    Ktval=Ktval+k(i,i);
    for j=i+1:n_train
        if y_train(i)==y_train(j)
            Ktval=Ktval+2*k(i,j);
        end
    end
end


Aeq=kron(ones(1,Y),eye(n_cal));
    beq=[ones(n_cal,1)];

   Aloss=double((1/n_cal)*(-log(sc_cal(:)+10^-40))');
 
  if mosek==0 
         options=optimoptions('quadprog','LinearSolver','auto','Algorithm','interior-point-convex','OptimalityTolerance', 1e-4, ...
    'ConstraintTolerance', 1e-4, ...
    'StepTolerance', 1e-6);

         lb=zeros(n_cal*Y,1);
         ub=ones(n_cal*Y,1);

tic
    p=quadprog(sparse(2*Kc/(n_cal^2)),sparse(-2*v/(n_train*n_cal)),sparse(Aloss),sparse(upper_loss),sparse(Aeq),sparse(beq),lb,ub,[],options);
time=toc;

  else
   

tic
cvx_solver mosek
cvx_begin quiet
variables p(n_cal*Y,1) 

minimize ((1/n_cal)*quad_form(p,Kc)-(1/(n_train))*2*v'*p)

p>=0;
Aeq*p==beq; 
Aloss*p<=double(upper_loss);
cvx_end 
time=toc;
  end

val= ((1/(n_cal^2))*p'*Kc*p-(1/(n_cal*n_train))*2*v'*p)+(1/(n_train^2))*Ktval;


