function q = weighted_quantile(values, probs, p)
    p=min(p,1);
    p=max(p,0);

    probs=abs(probs);
    probs=probs/sum(probs);

    [values_sorted, idx] = sort(values,'ascend');
    probs_sorted = probs(idx);

    cdf = cumsum(probs_sorted);
    

    cdf = cdf / cdf(end);
    
cum=probs_sorted(1);
 cont=1;
 while cum<p-10^-10
     cont=cont+1;
     cum=cum+probs_sorted(cont);
 end

 q=values_sorted(cont);


end