%% 8/26/2021
% Diagnostic facet status model
% The purpose is to generate propoer Q-matrix, and see if "fmincon" and handle
% high dimensional optimization

%clear all
clc

Kalpha=3; % 3 target attributes
Kbeta=5; % 5 problematic attributes
K=Kalpha+Kbeta;

% First, assume alpha and beta reside in the same space
% Many of the combinations are not plausible
trueprofile=zeros(2^K,K);
a=[0;1];
for k=1:K
    true1=repmat(a,2^(k-1),2^(K-k));
    trueprofile(:,k)=reshape(true1',2^K,1);
end
colsum=[sum(trueprofile(:,1:3),2),sum(trueprofile(:,4:8),2)];
%%
%%%%%%%%%%%%%%Condition 1
permissible=ones(2^K,1);
prior=zeros(2^K,1);
for l=1:2^K
    if (colsum(l,1)>=2 && colsum(l,2)>2)  % 25% sparsity
        permissible(l)=0;
    end
end
prior(permissible==1)=1/sum(permissible);

% %%%%%%%%%%%%%Condition 2
% permissible=ones(2^K,1);
% prior=zeros(2^K,1);
% for l=1:2^K
%     if (colsum(l,1)>=2 && colsum(l,2)>=2)  % ~40% sparsity
%         permissible(l)=0;
%     end
% end
% prior(permissible==1)=1/sum(permissible);
%%
%%%%%%%%%%%%%Another way of generating true priors
N=4000;
lambda0=unifrnd(-1,0.5,[1,K]);
lambda1=[0.9,1.0,1.1,-0.9,-0.95,-1,-1.05,-1.1].*1.5;
theta=normrnd(0,1,[N,1]);
truep=zeros(N,K);
for i=1:N
    prob=1./(1+exp(-1.7*lambda1.*(theta(i)-lambda0)));
    truep(i,:)=binornd(1,prob);
end
[aa,~,c] = unique(truep,'rows');
temprow = [aa, histcounts(c,1:max(c)+1)']; %unique rows in the generated true profile matrix
sparsity=1-length(temprow)/(2.^K);  %sparsity proportion
prior=zeros(2^K,1);
for l=1:2^K
    [q,idx]=ismember(trueprofile(l,:),temprow(:,1:8),'rows');
    if (q==1)
        prior(l)=temprow(idx,K+1)/N;
    end
end

%% Generate item Q-matrix
% Assume there are 27 items, each has 4 response options.
% Among them, 9 items: key requries only one of the target attributes
% (forming 3 identity matrices) and none of the distractors require any
% target attributes; 9 items: key requires two of the target attributes,
% whereas 1 distractor requires one of the target attributes, all distractors requires only 1 of the problematic attributes; 
% 9 items with key requiring 2 attributes, one distractor requires 2 of the
% problematic attributes, and 1 distractor requires one of the target attributes
%% Generate item parameters (consider additive CDM)
qmatrix=table2array(Simulationtrue(:,3:10));
truekey=table2array(SimulationtrueS1(:,2));
L=27;
lambda0=unifrnd(-1,1,[L,1]);
%%
J1=zeros(L,1); %number of goal facets
J2=zeros(L,1); %number of problematic facets
lambda1=zeros(L,3); % slope in front of goal facets 
lambda2=zeros(L,5); % slope in front of problematic facets
lambda1_indi=zeros(L,3); % indicator that implies which elements of lambda 1 are non-zero
lambda2_indi=zeros(L,5);
for j=1:L
    qj=qmatrix(((j-1)*4+1):(4*j), :);
    J1(j)=sum(qj(truekey(j),1:3));  % total number of goal facets=the number of goal facets measured by the key
    J2(j)=length(find(sum(qj(:,4:8))>0)); % total number of problematic facets
    % considering that some problematic facets may be measured by more than
    % 1 distractors, I first compute column sums, and find non-zero
    % elements
    lambda1_indi(j,:)=qj(truekey(j),1:3)>0;
    lambda2_indi(j,:)=sum(qj(:,4:8))>0; 
end
%%
% sum(lambda1_indi)
% 15 instead of 21 shown in the excel, because among the last 18 items, one
% distractor of each item measures 1 of the goal facets, adding up to
% 18*1/3=6, and 15+6=21

% column sum from excel from problematic facets =18 which comes from:
% (27*3+9)/5=18
%%
lambda_1=unifrnd(1.75, 2.25, [sum(J1),1]);
count0=1;
for j=1:L
    count1=count0+sum(lambda1_indi(j,:))-1;
    temp=find(lambda1_indi(j,:)>0);
    lambda1(j,temp)=lambda_1(count0:count1);
    count0=count1+1;
end
%
lambda_2=unifrnd(1.75, 2.25, [sum(J2),1]);
count0=1;
for j=1:L
    count1=count0+sum(lambda2_indi(j,:))-1;
    temp=find(lambda2_indi(j,:)>0);
    lambda2(j,temp)=lambda_2(count0:count1);
    count0=count1+1;
end

%% Compute probability
% true_profile=truep(1,:);
% 
% exp_prob=zeros(L, 4);
% resp_prob=zeros(L,4);  % the probability of endorsing each category
% response_gen=zeros(L,1);
% for j=1:L
%     qj=qmatrix(((j-1)*4+1):(4*j), :);
%     for r=1:4
%         if (truekey(j)==r)
%             exp_prob(j,r)=1;
%         else
%             temp1=true_profile(1:3).*(qj(r,1:3)-qj(truekey(j),1:3));
%             temp2=true_profile(4:8).*qj(r,4:8);
%             exp_prob(j,r)=exp(lambda0(j)+lambda1(j,:)*temp1'+lambda2(j,:)*temp2');
%         end
%     end
%     resp_prob(j,:)=exp_prob(j,:)./sum(exp_prob(j,:)); 
%     response_gen(j)=find(mnrnd(1,resp_prob(j,:))==1);
% end

%% Try to estimate one item parameter
response=zeros(N,L);
response_prob=zeros(N,L*4);
for i=1:N
    for j=1:L
         qj=qmatrix(((j-1)*4+1):(4*j), :);
         [response(i,j),response_prob(i,((j-1)*4+1):(4*j))]=DFSM_response(truekey(j),lambda0(j),lambda1(j,:),lambda2(j,:),qj,truep(i,:));
    end
end
%%
% Given known "truep" (true profile), estimate item paraemters
% This serves as a benchmark
bias_intercept=0;
bias_slope=0;
rmse_slope=0;
rmse_intercept=0;
for j=1:L
    qj=qmatrix(((j-1)*4+1):(4*j), :);
    plength=J1(j)+J2(j)+1;
    initial=[0,ones(1, plength-1)];
    A=[];
    b=[];
    Aeq=[];
    beq=[];
    lb=[-2,zeros(1,plength-1)];
    ub=[2,4.*ones(1,plength-1)];
    output=fmincon(@(temp_para)DFSM_itemnlikelihood(response(:,j),truep,qj,truekey(j),N,lambda1_indi(j,:),lambda2_indi(j,:),J1(j),J2(j),temp_para),initial,A,b,Aeq,beq,lb,ub);
    truepara=[lambda0(j),lambda1(j,lambda1_indi(j,:)>0),lambda2(j,lambda2_indi(j,:)>0)];
    bias_intercept=bias_intercept+(output(1)-truepara(1));
    rmse_intercept=rmse_intercept+(output(1)-truepara(1)).^2;
    bias_slope=bias_slope+mean(output(2:(plength-1))-truepara(2:(plength-1)));
    rmse_slope=rmse_slope+mean((output(2:(plength-1))-truepara(2:(plength-1))).^2);
end