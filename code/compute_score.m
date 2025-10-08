function score=compute_score(probs,y)

[probs_sort,pi]=sort(probs,'descend');



    score=probs_sort(1);
   
cont=1;
    while pi(cont)~=y
        cont=cont+1;
        score=score+probs_sort(cont);
    end
u=unifrnd(0,1);
score=score-u*probs_sort(cont); 


end
