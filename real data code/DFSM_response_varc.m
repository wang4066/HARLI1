function [response, responseprob]=DFSM_response_varc(truekeyj,lambda0j,lambda1j,lambda2j,qj,true_profilei, num_responsej)
% truekeyj, lambda0j,lambda1j,qj,true_profilei refers to item parameters
% for item j and person parameters for person i
exp_probj=zeros(num_responsej,1);
for r=1:num_responsej
    if (truekeyj==r)
        exp_probj(r)=1;
    else
        temp1=true_profilei(1:3).*(qj(r,1:3)-qj(truekeyj,1:3));
        temp2=true_profilei(4:8).*qj(r,4:8);
        exp_probj(r)=exp(lambda0j+lambda1j*temp1'+lambda2j*temp2');
    end
end
resp_probj=exp_probj./sum(exp_probj);
response=find(mnrnd(1,resp_probj)==1);
responseprob=resp_probj';