function [estprofile, profile_prob]=DFSM_MAP(responsei,qmatrix,truekey,L,lambda0,lambda1,lambda2,prior,trueprofile, Kalpha, Kbeta)


%estprofile=zeros(1,Kalpha+Kbeta); % row vector
log_likelihood=zeros(2^(Kalpha+Kbeta),1); % column vector

for j=1:L
    truekeyj=truekey(j);
    qj=qmatrix(((j-1)*4+1):(4*j), :);
    lambda0j=lambda0(j);
    lambda1j=lambda1(j,:);
    lambda2j=lambda2(j,:);
    for l=1:2^(Kalpha+Kbeta)
        temp_profile=trueprofile(l,:);
        exp_probj=zeros(4,1);

        for r=1:4
            if (truekeyj==r)
                exp_probj(r)=1;
            else
                temp1=temp_profile(1:3).*(qj(r,1:3)-qj(truekeyj,1:3));
                temp2=temp_profile(4:8).*qj(r,4:8);
                exp_probj(r)=exp(lambda0j+lambda1j*temp1'+lambda2j*temp2');
            end
        end
        resp_probj=exp_probj./sum(exp_probj);
        log_likelihood(l)=log_likelihood(l)+log(resp_probj(responsei(j)));
    end
end

temp=exp(log_likelihood).*prior;
profile_prob=temp./sum(temp);
[~,I] = max(profile_prob);
estprofile=trueprofile(I,:);

