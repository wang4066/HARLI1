%% Summary of additional simulation results
% 3/3/2024
TNR=0;
TPR=0;
FDR1=0;
overall_profile=0;
overall_facet=0;
alpha_profile=0;
alpha_facet=0;
beta_profile=0;
beta_facet=0;
lambda0result=zeros(1,3); % 1: mean bias 2: relative bias 3: absolute bias
lambda1result=zeros(1,3);
lambda2result=zeros(1,3);

for rep=1:10
    %filename = ['DFSM_full_hierarchical_N2000_' int2str(rep) '.mat'];
    filename = ['DFSM_full_moderatecorr_N2000_' int2str(rep) '.mat'];
    load(filename)
%%%%%%%%%%%Table 6
    bestsolution=find(ebic(:,1)==min(ebic(:,1)));
    %ebic results
    %1-BIC, 2_AIC, 3_error rate, 4_mean ABC,5_correct recovery of true o
    % (TNR)
    % 6_correct recovery of true non-zeros (TPR), 7_(1-FDR), 8_-2*log-marginal
    TNR=TNR+ebic(bestsolution,5);
    TPR=TPR+ebic(bestsolution,6);
    FDR1=FDR1+ebic(bestsolution, 7);


    overall_profile=overall_profile+facetprofilerecovery;
    overall_facet=overall_facet+facetrecovery;


%%%%%%%%%%%%Table 5
  
    alpha_attribute=zeros(Kalpha,1);
    for k=1:Kalpha
        alpha_attribute(k)=sum(estimated_profile(1:N,k)==truep(1:N,k))/N;
    end
    alphaattribute=mean(alpha_attribute);
    alpha_facet=alpha_facet+alphaattribute;

    count=0;
    for i=1:N
        if (sum(abs(estimated_profile(i,1:Kalpha)-truep(i,1:Kalpha)))==0)
            count=count+1;
        end
    end
    alpha_profile=alpha_profile+count/N;

    count=0;
    for i=1:N
        if (sum(abs(estimated_profile(i,(Kalpha+1):(Kalpha+Kbeta))-truep(i,(Kalpha+1):(Kalpha+Kbeta))))==0)
            count=count+1;
        end
    end
    beta_profile=beta_profile+count/N;

    beta_attribute=zeros(Kalpha,1);
    for k=1:Kbeta
        beta_attribute(k)=sum(estimated_profile(1:N,k+Kalpha)==truep(1:N,k+Kalpha))/N;
    end
    betaattribute=mean(beta_attribute);
    beta_facet=beta_facet+betaattribute;

    %%%%%%%%%%%%Table 4
    lambda0result=lambda0result+[biaslambda0(bestsolution,1), relative_biaslambda0(bestsolution,1), absbiaslambda0(bestsolution,1)];
    temp1=mean(biaslambda1(bestsolution,:));
    temp2=mean(relative_biaslambda1(bestsolution,:));
    temp3=mean(absbiaslambda1(bestsolution,:));
    lambda1result=lambda1result+[temp1,temp2,temp3];

    temp1=mean(biaslambda2(bestsolution,:));
    temp2=mean(relative_biaslambda2(bestsolution,:));
    temp3=mean(absbiaslambda2(bestsolution,:));
    lambda2result=lambda2result+[temp1,temp2,temp3];
end


Replication=10;
TNR=TNR/Replication;
TPR=TPR/Replication;
FDR1=FDR1/Replication;
overall_profile=overall_profile/Replication;
overall_facet=overall_facet/Replication;
alpha_profile=alpha_profile/Replication;
alpha_facet=alpha_facet/Replication;
beta_profile=beta_profile/Replication;
beta_facet=beta_facet/Replication;
lambda0result=lambda0result./Replication; % 1: mean bias 2: relative bias 3: absolute bias
lambda1result=lambda1result./Replication;
lambda2result=lambda2result./Replication;


%%%%%%%%%%%%%%%%%Item Fit %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all
type_1=zeros(25,5);
% for replication 14~25, it contains a "type1" variable
for rep=1:10
    filename = ['DFSM_full_hierarchical_N2000_' int2str(rep) '.mat'];
    %filename = ['DFSM_full_moderatecorr_N2000_' int2str(rep) '.mat'];
    load(filename)
    [aa,~,c] = unique(estimated_profile,'rows');
    % "aa" denotes unique rows, c is a N-by-1 vector denoting the location
    % of each estimated profile relative to the unique profile
    temprow = [aa, histcounts(c,1:max(c)+1)']; %unique rows in the generated true profile matrix, and the last column is the count
    group=length(aa); %unique number of facet profiles
    item_chisq=zeros(L,5); % 1: Chi-square, 2: df, 3: p-value
    num_response=4.*ones(L,1);
    count=1;
    for j=1:L
        for g=1:group
            group_indicator=find(c==g);
            % observed count of people in each resposne category of item i
            qj=qmatrix(count:(count+num_response(j)-1), :);
            temp_response=response(group_indicator, j);
            observed_count=histcounts(temp_response)./temprow(g,9);
            [temp,expected_prob]=DFSM_response(truekey(j),est_lambda0(j),est_lambda1(j,:),est_lambda2(j,:),qj,aa(g,:));
            expected_count=expected_prob;
            %[observed_count;expected_count]
            if (length(observed_count)==num_response(j) && min(observed_count.*temprow(g,9))>=5)
                [j,g];
                [observed_count;expected_count];
                numerator=(observed_count-expected_count).^2;
                denominator=expected_count.*(1-expected_count);
                item_chisq(j,1)=item_chisq(j,1)+sum(numerator./denominator)*temprow(g,9);
            end
        end
        item_chisq(j,4)=group-sum(lambda2_indi(j,:))-sum(lambda1_indi(j,:))-1; % for df, this definition will yeild much inflated type i error
        item_chisq(j,2)=group*(num_response(j)-1)-sum(lambda2_indi(j,:))-sum(lambda1_indi(j,:))-1;
        item_chisq(j,3)=chi2cdf(item_chisq(j,1), item_chisq(j,2),'upper');
        item_chisq(j,5)=chi2cdf(item_chisq(j,1), item_chisq(j,4),'upper');
        count=count+num_response(j);
    end
    %filename1 = ['DFSM_full_moderatecorr_N2000_' int2str(rep) '.mat'];
    %save(filename1)
    type_1(rep,1)=rep;
    type_1(rep,2)=length(find(item_chisq(:,3)<.05/27)); % large df, lower Type 1
    type_1(rep,3)=length(find(item_chisq(:,5)<.05/27));
    type_1(rep,4)=length(find(item_chisq(:,3)<.05));
    type_1(rep,5)=length(find(item_chisq(:,5)<.05));
end
%[mean(type_1(11:25,2)./27),mean(type_1(11:25,3)./27),mean(type_1(11:25,4)./27),mean(type_1(11:25,5)./27)]
[mean(type_1(:,2)./27),mean(type_1(:,3)./27),mean(type_1(:,4)./27),mean(type_1(:,5)./27)]




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% based on discussion with Jimmy (3/7/2024)
temprowj=temprow(:,[1,4,5,6]);
[aaj,~,cj]=unique(temprowj,'rows');
length(aaj);
% if length(aaj)<length(aa), it implies that further collapsing is needed
type_1=zeros(25,3);


%%
clear all
type_1=zeros(25,5);
for rep=11:25
    filename = ['DFSM_full_hierarchical_N2000_' int2str(rep) '.mat'];
    %filename = ['DFSM_full_moderatecorr_N2000_' int2str(rep) '.mat'];
    load(filename)
    %%%%%%%%3/7/24%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Following Jimmy de la Torre's suggestion

%    [aa,~,c] = unique(estimated_profile,'rows');
    % "aa" denotes unique rows, c is a N-by-1 vector denoting the location
    % of each estimated profile relative to the unique profile
%    temprow = [aa, histcounts(c,1:max(c)+1)']; %unique rows in the generated true profile matrix, and the last column is the count
%    groupj=length(aa); %unique number of facet profiles
    item_chisq_jimmy=zeros(L,6); % 1: Chi-square, 2: df, 3: p-value; 4: another df, 5. another pvalue 6. # of permissible groups
    num_response=4.*ones(L,1);
    count=1;
    for j=1:L
       qj=qmatrix(count:(count+num_response(j)-1), :);
       ir_facet=find(sum(qj)==0); % irrelvant facet for item j
       reduced_profilej=estimated_profile;
       reduced_profilej(:,ir_facet)=[];
       [aa,~,c] = unique(reduced_profilej,'rows');
       group=length(aa);

       temprowj = [aa, histcounts(c,1:max(c)+1)'];
       lastc=size(temprowj,2);
        for g=1:group
            group_indicator=find(c==g);
            % observed count of people in each resposne category of item i
            
            temp_response=response(group_indicator, j);
            observed_count=histcounts(temp_response)./temprowj(g,lastc);

            uniqg=estimated_profile(group_indicator(1),:); % find any profile that belongs to this group (they will all give the same expected probability)
            [temp,expected_prob]=DFSM_response(truekey(j),est_lambda0(j),est_lambda1(j,:),est_lambda2(j,:),qj,uniqg);
            expected_count=expected_prob;
            %[observed_count;expected_count]
            if (length(observed_count)==num_response(j) && min(observed_count.*temprowj(g,lastc))>=5)
                [j,g]
                [observed_count;expected_count]
                numerator=(observed_count-expected_count).^2;
                denominator=expected_count.*(1-expected_count);
                item_chisq_jimmy(j,1)=item_chisq_jimmy(j,1)+sum(numerator./denominator)*temprowj(g,lastc);
            end
        end
        item_chisq_jimmy(j,4)=group-sum(lambda2_indi(j,:))-sum(lambda1_indi(j,:))-1; % for df, this definition will yeild much inflated type i error
        item_chisq_jimmy(j,2)=group*(num_response(j)-1)-sum(lambda2_indi(j,:))-sum(lambda1_indi(j,:))-1;
        item_chisq_jimmy(j,3)=chi2cdf(item_chisq_jimmy(j,1), item_chisq_jimmy(j,2),'upper');
        item_chisq_jimmy(j,5)=chi2cdf(item_chisq_jimmy(j,1), item_chisq_jimmy(j,4),'upper');
        item_chisq_jimmy(j,6)=group;
        count=count+num_response(j);
    end
    type_1(rep,1)=rep;
    type_1(rep,2)=length(find(item_chisq_jimmy(:,3)<.05/27));  % much larger df, hence lower Type I error
    type_1(rep,3)=length(find(item_chisq_jimmy(:,5)<.05/27));
    type_1(rep,4)=length(find(item_chisq_jimmy(:,3)<.05));  % much larger df, hence lower Type I error
    type_1(rep,5)=length(find(item_chisq_jimmy(:,5)<.05));
end
%[mean(type_1(11:25,2)./27), mean(type_1(11:25,3)./27),mean(type_1(11:25,4)./27), mean(type_1(11:25,5)./27)]
[mean(type_1(:,2)./27),mean(type_1(:,3)./27),mean(type_1(:,4)./27),mean(type_1(:,5)./27)]