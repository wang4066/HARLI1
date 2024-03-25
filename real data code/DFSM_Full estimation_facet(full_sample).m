%% 3/13/2023 Analyze real facet data
% Revised in March 2-024, used the full sample to obtain final parameters
% and item fit results

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Facet data check###############
Kalpha=3;
Kbeta=5;
L=9; % 9 items

beta=table2array(qmatrixcombined(:,3:7));
alpha=table2array(qmatrixcombined(:,8:10));
qmatrix=[alpha,beta];
truekey=table2array(keyanswerforIdentifyingForces)';
response_full=table2array(response1);
N=length(response_full);

%%
%for bootstrap_rep=5:10
    
    response=response_full;

    %N=2000;
    %Ntest=length(response_full)-N;

%     response_fullcopy=response_full;
%     response_fullcopy(c,:)=[];
%     response_test=response_fullcopy; % create a response data set for test sample

    lambda1_indi=zeros(L,Kalpha); % indicator that implies which elements of lambda 1 are non-zero
    lambda2_indi=zeros(L,Kbeta);
    num_response=[4, 4, 5, 4, 4, 4, 4, 4, 4]';
    count=1;
    for j=1:L
        qj=qmatrix(count:(count+num_response(j)-1), :);
        J1(j)=sum(qj(truekey(j),1:3));  % total number of goal facets=the number of goal facets measured by the key
        J2(j)=length(find(sum(qj(:,4:8))>0)); % total number of problematic facets
        % considering that some problematic facets may be measured by more than
        % 1 distractors, I first compute column sums, and find non-zero
        % elements
        lambda1_indi(j,:)=qj(truekey(j),1:Kalpha)>0;
        lambda2_indi(j,:)=sum(qj(:,(Kalpha+1):(Kalpha+Kbeta)))>0;
        count=count+num_response(j);
    end

    K=Kalpha+Kbeta;
    trueprofile=zeros(2^K,K);
    a=[0;1];
    for k=1:K
        true1=repmat(a,2^(k-1),2^(K-k));
        trueprofile(:,k)=reshape(true1',2^K,1);
    end

    pgamma=-2.25:0.15:-1.5; % smaller pgamma induces more sparsity
    C=0.00001;
    rhoN=1/(N^2); % a threshold according to Gu & Xu
    Accuracy_of_Iteration=0.025;
    ebic=zeros(length(pgamma),8);%1-BIC, 2_AIC, 3_error rate, 4_mean ABC,5_correct recovery of true o
    % 6_correct recovery of true non-zeros, 7_(1-FDR), 8_-2*log-marginal
    Maximum_EM_Cycles=200;
    interimcheck=zeros(length(pgamma),2);
    solutionpath=zeros((2^K+1), length(pgamma)); % last row is to indicate the # of zero elements (i.e., sparsity)

    % use true item parameters as starting values (a bit cheating for now)



    initial_lambda0=zeros(L,1);
    initial_lambda1=zeros(L,Kalpha);
    initial_lambda1(lambda1_indi==1)=2;
    initial_lambda2=zeros(L,Kbeta);
    initial_lambda2(lambda2_indi==1)=2;

    for penalty=1:length(pgamma)

        pi_initial=1/(2^K).*ones(1,2^K); % uniform distribution as initial values
        Number_of_EM_Cycles=0;
        flag1=1;
        while (flag1==1)
            Number_of_EM_Cycles=Number_of_EM_Cycles+1;

            if (Number_of_EM_Cycles>Maximum_EM_Cycles)
                break;
            end


            likelihoodmatrix=zeros(N,2^K);
            for i=1:N
                temp=DFSM_likelihood_varc(response(i,:),qmatrix,truekey,L,initial_lambda0,initial_lambda1,initial_lambda2,trueprofile,Kalpha,Kbeta,num_response);
                likelihoodmatrix(i,:)=temp';
            end

            % compute posterior
            posterior=zeros(N,2^K);
            for i=1:N
                temp=likelihoodmatrix(i,:).*pi_initial;
                posterior(i,:)=temp./sum(temp);
            end

            deltal=max(C,(pgamma(penalty)+sum(posterior,1))); %1-by-2^K vector
            pi_final=deltal./sum(deltal);
            est_lambda0=zeros(L,1);
            est_lambda1=zeros(L,Kalpha);
            est_lambda2=zeros(L,Kbeta);

            count=1;
            for j=1:L
                %qj=qmatrix(((j-1)*4+1):(4*j), :);
                qj=qmatrix(count:(count+num_response(j)-1), :);
                plength=J1(j)+J2(j)+1;
                initial=[0,ones(1, plength-1)]; % the initial values could be more informative later
                A=[];
                b=[];
                Aeq=[];
                beq=[];
                lb=[-2,zeros(1,plength-1)];
                ub=[2,4.*ones(1,plength-1)];
                output=fmincon(@(temp_para)DFSM_itemnlikelihood_unknown_varc(response(:,j),posterior,qj,truekey(j),N,lambda1_indi(j,:),lambda2_indi(j,:),J1(j),J2(j),temp_para,K,trueprofile,num_response(j)),initial,A,b,Aeq,beq,lb,ub);

                est_lambda0(j)=output(1);
                %temp=find(lambda1_indi(j,:)==1); % find locations of non-zero lambda1 entries
                est_lambda1(j,(lambda1_indi(j,:)==1))=output(2:(1+J1(j)));
                %temp=find(lambda2_indi(j,:)==1); % find locations of non-zero lambda1 entries
                est_lambda2(j,(lambda2_indi(j,:)==1))=output((2+J1(j)):plength);
                count=count+num_response(j);
            end

            dif1=max(max(abs(est_lambda1-initial_lambda1)));
            dif2=max(max(abs(est_lambda2-initial_lambda2)));
            dif3=max(abs(est_lambda0-initial_lambda0));
            dif4=max(abs(pi_final-pi_initial));

            if (max([dif1,dif2,dif3,dif4])<Accuracy_of_Iteration)
                flag1=0;
            else
                pi_initial=pi_final;
                initial_lambda0=est_lambda0;
                initial_lambda1=est_lambda1;
                initial_lambda2=est_lambda2;

            end
        end
        pi_final_p=max(rhoN.*ones(1,2^K),pi_final);
        pi_final_p(pi_final_p==rhoN)=0;
        interimcheck(penalty,1)=sum(pi_final_p==0);
        interimcheck(penalty,2)=Number_of_EM_Cycles;

        % After deciding on the sparsity structure of tau12, re-run EM to
        % obtain EBIC
        pi_initial=1/(2^K).*ones(1,2^K);
        pi_initial(pi_final_p==0)=0;
        pi_initial=pi_initial./(sum(pi_initial)); %standardization so that it still sums up to 1

        Number_of_EM_Cycles=0;
        flag1=1;
        while (flag1==1)
            Number_of_EM_Cycles=Number_of_EM_Cycles+1;

            if (Number_of_EM_Cycles>Maximum_EM_Cycles)
                break;
            end

            likelihoodmatrix=zeros(N,2^K);
            for i=1:N
                temp=DFSM_likelihood_varc(response(i,:),qmatrix,truekey,L,initial_lambda0,initial_lambda1,initial_lambda2,trueprofile,Kalpha,Kbeta,num_response);
                likelihoodmatrix(i,:)=temp';
            end


            % compute posterior
            posterior=zeros(N,2^K);
            for i=1:N
                temp=likelihoodmatrix(i,:).*pi_initial;
                posterior(i,:)=temp./sum(temp);
            end

            deltal=sum(posterior);
            pi_final=deltal./sum(deltal);
            est_lambda0=zeros(L,1);
            est_lambda1=zeros(L,Kalpha);
            est_lambda2=zeros(L,Kbeta);
            count=1;
            for j=1:L
                %qj=qmatrix(((j-1)*4+1):(4*j), :);
                qj=qmatrix(count:(count+num_response(j)-1), :);
                plength=J1(j)+J2(j)+1;
                initial=[0,ones(1, plength-1)]; % the initial values could be more informative later
                A=[];
                b=[];
                Aeq=[];
                beq=[];
                lb=[-2,zeros(1,plength-1)];
                ub=[2,4.*ones(1,plength-1)];
                output=fmincon(@(temp_para)DFSM_itemnlikelihood_unknown_varc(response(:,j),posterior,qj,truekey(j),N,lambda1_indi(j,:),lambda2_indi(j,:),J1(j),J2(j),temp_para,K,trueprofile,num_response(j)),initial,A,b,Aeq,beq,lb,ub);

                est_lambda0(j)=output(1);
                %temp=find(lambda1_indi(j,:)==1); % find locations of non-zero lambda1 entries
                est_lambda1(j,(lambda1_indi(j,:)==1))=output(2:(1+J1(j)));
                %temp=find(lambda2_indi(j,:)==1); % find locations of non-zero lambda1 entries
                est_lambda2(j,(lambda2_indi(j,:)==1))=output((2+J1(j)):plength);
                count=count+num_response(j);
            end

            dif1=max(max(abs(est_lambda1-initial_lambda1)));
            dif2=max(max(abs(est_lambda2-initial_lambda2)));
            dif3=max(abs(est_lambda0-initial_lambda0));
            dif4=max(abs(pi_final-pi_initial));

            if (max([dif1,dif2,dif3,dif4])<Accuracy_of_Iteration)
                flag1=0;
            else
                pi_initial=pi_final;
                initial_lambda0=est_lambda0;
                initial_lambda1=est_lambda1;
                initial_lambda2=est_lambda2;

            end
        end

       

        %***************
        % compute BIC
        marginal_log=0.0;
        for i=1:N
            marginal_log=marginal_log+log(sum(pi_final.*likelihoodmatrix(i,:)));
        end
        count=sum(pi_final>0);
        ebic(penalty,1)=-2*marginal_log+log(N)*count; % BIC
        ebic(penalty,2)=-2*marginal_log+2*count; % AIC
        solutionpath(1:2^K,penalty)=pi_final;
        solutionpath((2^K+1),penalty)=sum(pi_final==0);
        est_sparsity=(pi_final==0);
        %     true_sparsity=(prior==0)';
        %     ebic(penalty,3)=sum(abs(est_sparsity-true_sparsity))/(2^K); %error rate of the sparsity
        %     ebic(penalty,4)=mean(abs(pi_final'-prior)); % mean absolute bias
        %     % proportion of true 0's that are correctly recovered
        %     est_sparsity=find(pi_final==0);
        %     true_sparsity=find(prior==0);
        %     ebic(penalty,5)=length(intersect(est_sparsity, true_sparsity))/sum(prior==0);
        %     % proportion of true non-zero's that are correctly recovered % TPR in
        %     % Gu & Xu (2019)
        %     est_sparsity1=find(pi_final>0);
        %     true_sparsity1=find(prior>0);
        %     ebic(penalty,6)=length(intersect(est_sparsity1, true_sparsity1))/sum(prior>0);
        %     % 1-FDR: proportion of selected patterns that are true patterns
        %     ebic(penalty,7)=length(intersect(est_sparsity1, true_sparsity1))/sum(pi_final>0);
        %     ebic(penalty,8)=marginal_log; % log-marginal likelihood, no penalty
        filename = ['DFSM_facet_full_sample.mat'];
        save(filename);
    end


    %%
    estimated_profile=zeros(N,Kalpha+Kbeta);
    est_profileprob=zeros(N,2^(Kalpha+Kbeta));
    equ_prior=ones(256,1).*(1/256);

    estimated_profile_equprior=zeros(N,Kalpha+Kbeta);
    est_profileprob_equprior=zeros(N,2^(Kalpha+Kbeta));
    % for i=1:N  % let's focus on the first 1000 people as the algorithm can be slow at the moment
    % [estprofile, profile_prob]=DFSM_MAP(response(i,:),qmatrix,truekey,L,est_lambda0,est_lambda1,est_lambda2,pi_final',trueprofile,Kalpha, Kbeta);
    % estimated_profile(i,:)=estprofile;
    % est_profileprob(i,:)=profile_prob';
    %
    % end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%Discover Facet Map%%%%%%%%%%%%%%%%%
    %load('DFSM_facet_split.mat')
    bestsolution=find(ebic(:,1)==min(ebic(:,1))); % based on BIC

    checkpattern=[trueprofile,solutionpath(1:256,bestsolution)];
    checkpattern_reverse=checkpattern; % reverse coding problematic facets
    checkpattern_reverse(:,4:8)=1-checkpattern(:,4:8);
    permissiblelist=find(checkpattern_reverse(:,9)>0);
    perm_pattern_reverse=checkpattern_reverse(permissiblelist,:);

    % Check the relationship among alphas
    unique(perm_pattern_reverse(:,1:3),'rows'); % seems that they can be independent

    % alpha1, beta1
    unique(perm_pattern_reverse(:,[1,4]),'rows');
    % alpha1, beta2
    unique(perm_pattern_reverse(:,[1,5]),'rows');

    % Check pairs of facets
    hier_c=zeros(64,3+12);
    count=0;
    for i=1:7
        for j=(i+1):8
            count=count+1;
            hier_c(count,1)=i;
            hier_c(count,2)=j;
            hier_c(count,3)=length(unique(perm_pattern_reverse(:,[i,j]),'rows'));

            [C, ia, ic]=unique(perm_pattern_reverse(:,[i,j]),'rows');
            for k=1:hier_c(count,3)
                indicator=find(ic==k);
                hier_c(count, (3+(k-1)*3+1):(3+3*(k-1)+2))=C(k,:);
                hier_c(count, (3+3*k))=sum(perm_pattern_reverse(indicator,9));
            end
        end
    end

    % Check triplets of facets
    hier_c3=zeros(64,4+32);% 32=4 (profile of 3 facets+ its probabiltiy)* 8 possible patterns without link)
    count=0;
    for i=1:6
        for j=(i+1):7
            for k=(j+1):8
                count=count+1;
                hier_c3(count,1)=i;
                hier_c3(count,2)=j;
                hier_c3(count,3)=k;
                hier_c3(count,4)=length(unique(perm_pattern_reverse(:,[i,j,k]),'rows'));


                [C, ia, ic]=unique(perm_pattern_reverse(:,[i,j,k]),'rows');
                for l=1:hier_c3(count,4)
                    indicator=find(ic==l);
                    hier_c3(count, (4+(l-1)*4+1):(4+4*(l-1)+3))=C(l,:);
                    hier_c3(count, (4+4*l))=sum(perm_pattern_reverse(indicator,9));
                end
            end
        end
    end




    %% Re-run EM to obtain optimal item parameters so that we can compute mastery profile
    % In previous run, we don't save all item parameters per tuning parameter
    % but our solution path is useful to definie pi_final_p

    % After deciding on the sparsity structure of tau12, re-run EM to
    % obtain EBIC
    pi_initial=1/(2^K).*ones(1,2^K);
    %pi_initial(pi_final_p==0)=0;
    pi_initial(solutionpath(1:2^K,bestsolution)==0)=0; % we consider the best solution
    pi_initial=pi_initial./(sum(pi_initial)); %standardization so that it still sums up to 1

    Number_of_EM_Cycles=0;
    flag1=1;
    while (flag1==1)
        Number_of_EM_Cycles=Number_of_EM_Cycles+1;

        if (Number_of_EM_Cycles>Maximum_EM_Cycles)
            break;
        end

        likelihoodmatrix=zeros(N,2^K);
        for i=1:N
            temp=DFSM_likelihood_varc(response(i,:),qmatrix,truekey,L,initial_lambda0,initial_lambda1,initial_lambda2,trueprofile,Kalpha,Kbeta,num_response);
            likelihoodmatrix(i,:)=temp';
        end


        % compute posterior
        posterior=zeros(N,2^K);
        for i=1:N
            temp=likelihoodmatrix(i,:).*pi_initial;
            posterior(i,:)=temp./sum(temp);
        end

        deltal=sum(posterior);
        pi_final=deltal./sum(deltal);
        est_lambda0=zeros(L,1);
        est_lambda1=zeros(L,Kalpha);
        est_lambda2=zeros(L,Kbeta);
        count=1;
        for j=1:L
            %qj=qmatrix(((j-1)*4+1):(4*j), :);
            qj=qmatrix(count:(count+num_response(j)-1), :);
            plength=J1(j)+J2(j)+1;
            initial=[0,ones(1, plength-1)]; % the initial values could be more informative later
            A=[];
            b=[];
            Aeq=[];
            beq=[];
            lb=[-2,zeros(1,plength-1)];
            ub=[2,4.*ones(1,plength-1)];
            output=fmincon(@(temp_para)DFSM_itemnlikelihood_unknown_varc(response(:,j),posterior,qj,truekey(j),N,lambda1_indi(j,:),lambda2_indi(j,:),J1(j),J2(j),temp_para,K,trueprofile,num_response(j)),initial,A,b,Aeq,beq,lb,ub);

            est_lambda0(j)=output(1);
            %temp=find(lambda1_indi(j,:)==1); % find locations of non-zero lambda1 entries
            est_lambda1(j,(lambda1_indi(j,:)==1))=output(2:(1+J1(j)));
            %temp=find(lambda2_indi(j,:)==1); % find locations of non-zero lambda1 entries
            est_lambda2(j,(lambda2_indi(j,:)==1))=output((2+J1(j)):plength);
            count=count+num_response(j);
        end

        dif1=max(max(abs(est_lambda1-initial_lambda1)));
        dif2=max(max(abs(est_lambda2-initial_lambda2)));
        dif3=max(abs(est_lambda0-initial_lambda0));
        dif4=max(abs(pi_final-pi_initial));

        if (max([dif1,dif2,dif3,dif4])<Accuracy_of_Iteration)
            flag1=0;
        else
            pi_initial=pi_final;
            initial_lambda0=est_lambda0;
            initial_lambda1=est_lambda1;
            initial_lambda2=est_lambda2;

        end
    end
   

    % estimate student facet profile
    estimated_profile=zeros(N,Kalpha+Kbeta);
    est_profileprob=zeros(N,2^(Kalpha+Kbeta));
    for i=1:N  % let's focus on the first 1000 people as the algorithm can be slow at the moment
        [estprofile, profile_prob]=DFSM_MAP_varc(response(i,:),qmatrix,truekey,L,est_lambda0,est_lambda1,est_lambda2,pi_final',trueprofile,Kalpha, Kbeta,num_response);
        estimated_profile(i,:)=estprofile;
        est_profileprob(i,:)=profile_prob';
    end
    % average mastery of each attribute
    sum(estimated_profile)/N


    [aa,~,c] = unique(estimated_profile,'rows');
    temprow = [aa, histcounts(c,1:max(c)+1)']; %unique rows in the generated true profile matrix
    sparsity=1-length(temprow)/(2.^K);  %sparsity proportion
    sample_prob=zeros(2^K,1);
    N=2729;
    for l=1:2^K
        [q,idx]=ismember(trueprofile(l,:),temprow(:,1:8),'rows');
        if (q==1)
            sample_prob(l)=temprow(idx,K+1)/N;
        end
    end

   %%%%%%% 2/18/2024%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   % compute reliability of the facet classification
   % compute marginal probability of having each facet
   % Check the training sample of 2000 students first
   %N=2000;
   marginal_facet=zeros(N, Kalpha+Kbeta);
   for i=1:N
       for k=1:(Kalpha+Kbeta)
            marginal_facet(i,k)=trueprofile(:, k)'*est_profileprob(i,:)';
       end 
   end
   contigencyK=zeros(2,2,K);
  % index=find(estimated_profile(:,k)==1);
   index=1:N;
   for k=1:K
       contigencyK(1,1,k)=mean(marginal_facet(index,k).^2);
       contigencyK(1,2,k)=mean(marginal_facet(index,k).*(1-marginal_facet(index,k)));
       contigencyK(2,1,k)=contigencyK(1,2,k);
       contigencyK(2,2,k)=mean((1-marginal_facet(index,k)).^2);
   end
 % From contigency table to tetrachoric correlation (using R)

 %%%%%%%%%%%2/19/2024%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 % Evaluate item level absolute fit%%%%%%%%%%%%%%%%%%%%

    [aa,~,c] = unique(estimated_profile,'rows');
    temprow = [aa, histcounts(c,1:max(c)+1)']; %unique rows in the generated true profile matrix, and the last column is the count
    
    group=length(aa); %unique number of facet profiles
    item_chisq=zeros(L,5); % 1: Chi-square, 2: df, 3: p-value
    count=1;
    for j=1:L
        for g=1:group
            group_indicator=find(c==g);
            % observed count of people in each resposne category of item i
            qj=qmatrix(count:(count+num_response(j)-1), :);
            temp_response=response(group_indicator, j);
            observed_count=histcounts(temp_response)./temprow(g,9);
            [temp,expected_prob]=DFSM_response_varc(truekey(j),est_lambda0(j),est_lambda1(j,:),est_lambda2(j,:),qj,aa(g,:),num_response(j));
            expected_count=expected_prob;
            %[observed_count;expected_count]
            if (length(observed_count)==num_response(j) & min(observed_count.*temprow(g,9))>=5)
            [j,g];
            [observed_count;expected_count];
            numerator=(observed_count-expected_count).^2;
            denominator=expected_count.*(1-expected_count);
            item_chisq(j,1)=item_chisq(j,1)+sum(numerator./denominator)*temprow(g,9);
            end
        end
        item_chisq(j,2)=group*(num_response(j)-1)-sum(lambda2_indi(j,:))-sum(lambda1_indi(j,:))-1;
        item_chisq(j,3)=chi2cdf(item_chisq(j,1), item_chisq(j,2),'upper');
        count=count+num_response(j);
        item_chisq(j,4)=group-sum(lambda2_indi(j,:))-sum(lambda1_indi(j,:))-1;
        item_chisq(j,5)=chi2cdf(item_chisq(j,1), item_chisq(j,4),'upper');
    end


 filename = ['DFSM_facet_full_sample.mat'];
 save(filename)
%end

%%%%%%%%%%3/13/2024%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Use the item fit index suggested by Jimmy

%type_1=zeros(25,5);

% Following Jimmy de la Torre's suggestion

%    [aa,~,c] = unique(estimated_profile,'rows');
% "aa" denotes unique rows, c is a N-by-1 vector denoting the location
% of each estimated profile relative to the unique profile
%    temprow = [aa, histcounts(c,1:max(c)+1)']; %unique rows in the generated true profile matrix, and the last column is the count
%    groupj=length(aa); %unique number of facet profiles
item_chisq_jimmy=zeros(L,6); % 1: Chi-square, 2: df, 3: p-value; 4: another df, 5. another pvalue 6. # of permissible groups

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
        [temp,expected_prob]=DFSM_response_varc(truekey(j),est_lambda0(j),est_lambda1(j,:),est_lambda2(j,:),qj,uniqg,num_response(j));
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


