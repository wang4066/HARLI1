function [likelihood]=DFSM_likelihood_varc(responsei,qmatrix,truekey,L,lambda0,lambda1,lambda2,trueprofile, Kalpha, Kbeta,num_response)

% updated on 3/13/23: allowing different items to have different number of
% response categories

K=Kalpha+Kbeta;

log_likelihood=zeros(2^K,1); % column vector



count=1;
for j=1:L
    truekeyj=truekey(j);
    %qj=qmatrix(((j-1)*4+1):(4*j), :);
    qj=qmatrix(count:(count+num_response(j)-1), :);
    lambda0j=lambda0(j);
    lambda1j=lambda1(j,:);
    lambda2j=lambda2(j,:);
    %for l=1:2^K
       % temp_profile=trueprofile(l,:);
        exp_probj=zeros(num_response(j),2^K);

        for r=1:num_response(j)
            if (truekeyj==r)
                exp_probj(r,:)=1;
            else
                temp1=trueprofile(:,1:3).*(qj(r,1:3)-qj(truekeyj,1:3));
                temp2=trueprofile(:,4:8).*qj(r,4:8);
                exp_probj(r,:)=exp(lambda0j+lambda1j*temp1'+lambda2j*temp2');
            end
        end
        resp_probj=exp_probj./sum(exp_probj);
        log_likelihood=log_likelihood+log(resp_probj(responsei(j),:))';
    %end
    count=count+num_response(j);
end

likelihood=exp(log_likelihood);