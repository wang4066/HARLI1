function f=DFSM_itemnlikelihood_unknown_varc(responsej,posterior,qj,truekeyj, N, lambda1indii,lambda2indii, J1j, J2j,temp_para,K,trueprofile,num_responsej)
% updated 3/13/23: allowing items to have different response categories
% This version is for unknown attribute patterns
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

 resp_prob=zeros(2^K,num_responsej); % 4 denotes 4 response options
 for l=1:2^K
        exp_prob=zeros(1,num_responsej);
        for r=1:num_responsej
            if (truekeyj==r)
                exp_prob(r)=1;
            else
                temp1=trueprofile(l,1:3).*(qj(r,1:3)-qj(truekeyj,1:3));
                temp2=trueprofile(l,4:8).*qj(r,4:8);
                exp_prob(r)=exp(temp_para(1)+lambda1j*temp1'+lambda2j*temp2');
            end
        end
        resp_prob(l,:)=exp_prob./sum(exp_prob);
 end
 resp_prob(resp_prob>.9999)=.9999;
 resp_prob(resp_prob<.00001)=.00001;


for i=1:N
    like=like+posterior(i,:)*log(resp_prob(:,responsej(i)));
end

% for i=1:N
%     for l=1:2^K
%         exp_prob=zeros(1,4);
%         for r=1:4
%             if (truekeyj==r)
%                 exp_prob(r)=1;
%             else
%                 temp1=trueprofile(l,1:3).*(qj(r,1:3)-qj(truekeyj,1:3));
%                 temp2=trueprofile(l,4:8).*qj(r,4:8);
%                 exp_prob(r)=exp(temp_para(1)+lambda1j*temp1'+lambda2j*temp2');
%             end
%         end
%         resp_prob=exp_prob./sum(exp_prob);
%         p=resp_prob(responsej(i));
%         if (p<0.00001)
%             p=0.00001;
%         end
%         if (p>0.99999)
%             p=0.99999;
%         end
%         like=like+log(p)*posterior(i,l); % this may not be right as the
%         %posterior is used mutliple times. Need to double check
%     end
% end
f=-like;
    




