function f=DFSM_itemnlikelihood_unknown_varc_option(responsej,posterior,qj,truekeyj, N, Kalpha, Kbeta,temp_para,K,trueprofile,num_responsej)
% updated 3/13/23: allowing items to have different response categories
% This version is for unknown attribute patterns
% temp_para=zeros(1,1+J1j+J2j); 
% temp_para is a row vector of the unknown paraemters for this item that needs to be estiamted

% Updated on 2/14/2024
% Allow item parameters to differ at option level
% Now temp_para is updated

total_slope=sum(sum(qj));
%temp_para=zeros(1, num_responsej-1+total_slope); % the first (num_responsej-1) refer to intercepts
% 
% lambda1j=zeros(1,3);
% lambda2j=zeros(1,5);

% %goal=length(find(lambda1indii>0));
% goal=J1j; % total number of goal facets
% lambda1j(lambda1indii>0)=temp_para(2:(1+goal));
% %problematic=length(find(lambda2indii>0));
% problematic=J2j;
% lambda2j(lambda2indii>0)=temp_para((goal+2):(1+goal+problematic));



like=0;

resp_prob=zeros(2^K,num_responsej); % 4 denotes 4 response options



 index_slope=qj>0;
 lambda_slope=qj;
 lambda_slope(index_slope)=temp_para(1,num_responsej:(num_responsej-1+total_slope));
 lambda1j=lambda_slope(:, 1:Kalpha);
 lambda2j=lambda_slope(:, (Kalpha+1):(Kalpha+Kbeta));

 for l=1:2^K
        exp_prob=zeros(1,num_responsej);
       
        thresholdcount=0;
        for r=1:num_responsej
            if (truekeyj==r)
                exp_prob(r)=1;
            else
                thresholdcount=thresholdcount+1;
                temp1=trueprofile(l,1:3).*(qj(r,1:3)-qj(truekeyj,1:3));
                temp2=trueprofile(l,4:8).*qj(r,4:8);
                exp_prob(r)=exp(temp_para(thresholdcount)+lambda1j(r,:)*temp1'+lambda2j(r,:)*temp2');
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
    




