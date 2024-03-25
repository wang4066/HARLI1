function [likelihood]=DFSM_likelihood_varc_option(responsei,qmatrix,truekey,L,lambda0,lambda1,lambda2,trueprofile, Kalpha, Kbeta,num_response)

% updated on 3/13/23: allowing different items to have different number of
% response categories (varc denotes variable categories)
% updated on 2/13/24: allowing option level lambda parameters

K=Kalpha+Kbeta;

log_likelihood=zeros(2^K,1); % column vector



count=1;
count1=1; % this is only for "intercept" parameters. Note that the intercept for "key" response is always 0, so this count1 will be less than "count"
for j=1:L
    truekeyj=truekey(j);
    %qj=qmatrix(((j-1)*4+1):(4*j), :);
    qj=qmatrix(count:(count+num_response(j)-1), :);
    lambda0j=lambda0(count1:(count1+num_response(j)-2));
    lambda1j=lambda1(count:(count+num_response(j)-1),:);
    lambda2j=lambda2(count:(count+num_response(j)-1),:);
    %for l=1:2^K
       % temp_profile=trueprofile(l,:);
        exp_probj=zeros(num_response(j),2^K);
        
        thresholdcount=0;
        for r=1:num_response(j)
            if (truekeyj==r)
                exp_probj(r,:)=1;
            else
                thresholdcount=thresholdcount+1;
                temp1=trueprofile(:,1:3).*(qj(r,1:3)-qj(truekeyj,1:3));
                temp2=trueprofile(:,4:8).*qj(r,4:8);
                exp_probj(r,:)=exp(lambda0j(thresholdcount)+lambda1j(r,:)*temp1'+lambda2j(r,:)*temp2');
            end
        end
        resp_probj=exp_probj./sum(exp_probj);
        log_likelihood=log_likelihood+log(resp_probj(responsei(j),:))';
    %end
    count=count+num_response(j);
    count1=count1+num_response(j)-1;
end

likelihood=exp(log_likelihood);