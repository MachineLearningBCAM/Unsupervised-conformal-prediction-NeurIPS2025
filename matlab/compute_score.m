function score=compute_score(probs,y,u)

[probs_sort,pi]=sort(probs,'descend');



    score=probs_sort(1);
   
cont=1;
    while pi(cont)~=y
        cont=cont+1;
        score=score+probs_sort(cont);
    end
if nargin<3
    u=unifrnd(0,1);
end
score=score-u*probs_sort(cont); 


end
