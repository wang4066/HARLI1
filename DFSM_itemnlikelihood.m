function f=DFSM_itemnlikelihood(responsej,truep,qj,truekeyj, N, lambda1indii,lambda2indii, J1j, J2j,temp_para)

% temp_para=zeros(1,1+J1j+J2j); 
% temp_para is a row vector of the unknown paraemters for this item that needs to be estiamted

lambda1j=zeros(1,3);
lambda2j=zeros(1,5);

%goal=length(find(lambda1indii>0));
goal=J1j;
lambda1j(lambda1indii>0)=temp_para(2:(1+goal));
%problematic=length(find(lambda2indii>0));
problematic=J2j;
lambda2j(lambda2indii>0)=temp_para((goal+2):(1+goal+problematic));
like=0;
for i=1:N
    exp_prob=zeros(1,4);
    for r=1:4
        if (truekeyj==r)
            exp_prob(r)=1;
        else
            temp1=truep(i,1:3).*(qj(r,1:3)-qj(truekeyj,1:3));
            temp2=truep(i,4:8).*qj(r,4:8);
            exp_prob(r)=exp(temp_para(1)+lambda1j*temp1'+lambda2j*temp2');
        end
    end
    resp_prob=exp_prob./sum(exp_prob);
    p=resp_prob(responsej(i));
    if (p<0.00001)
        p=0.00001;
    end
    if (p>0.99999)
        p=0.99999;
    end
    like=like+log(p);
end
f=-like;
    




