%% 2/21/24
% Re-run simulation with 2,000 sample size
% Moderate correlation condition

%% Read in different item parameters
temp=readtable('TrueQ_key.xlsx');
qmatrix=table2array(temp(:,3:10));
temp=readtable('TrueQ_key.xlsx','Sheet',2);
truekey=table2array(temp(:,2)); % second tab of "Simulation_true.xls"
L=27;


Kalpha=3; % 3 target attributes
Kbeta=5; % 5 problematic attributes
K=Kalpha+Kbeta;
N=2000;
trueprofile=zeros(2^K,K);
a=[0;1];
for k=1:K
    true1=repmat(a,2^(k-1),2^(K-k));
    trueprofile(:,k)=reshape(true1',2^K,1);
end

for rep=2:10

    %%%%%%%%%%%%%%%%%%%%%%%
    % Simulate true facets from higher-order CDM
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

    filename = ['trueitemparameter_' int2str(rep) '.xls'];

    temp=readtable(filename);
    lambda0=table2array(temp);
    temp=readtable(filename,'Sheet',2);
    lambda1=table2array(temp);
    temp=readtable(filename,'Sheet',3);
    lambda2=table2array(temp);

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

    response=zeros(N,L);
    response_prob=zeros(N,L*4);
    for i=1:N
        for j=1:L
            qj=qmatrix(((j-1)*4+1):(4*j), :);
            [response(i,j),response_prob(i,((j-1)*4+1):(4*j))]=DFSM_response(truekey(j),lambda0(j),lambda1(j,:),lambda2(j,:),qj,truep(i,:));
        end
    end




    pgamma=0:-0.15:-1.35; % smaller pgamma induces more sparsity
    C=0.00001;
    rhoN=1/(N^2); % a threshold according to Gu & Xu
    Accuracy_of_Iteration=0.01;
    ebic=zeros(length(pgamma),8);%1-BIC, 2_AIC, 3_error rate, 4_mean ABC,5_correct recovery of true o
    % 6_correct recovery of true non-zeros, 7_(1-FDR), 8_-2*log-marginal
    Maximum_EM_Cycles=200;
    interimcheck=zeros(length(pgamma),2);
    solutionpath=zeros((2^K+1), length(pgamma)); % last row is to indicate the # of zero elements (i.e., sparsity)

    % use true item parameters as starting values (a bit cheating for now)
    initial_lambda0=lambda0;
    initial_lambda1=lambda1;
    initial_lambda2=lambda2;
    biaslambda0=zeros(length(pgamma));
    biaslambda1=zeros(length(pgamma),Kalpha);
    biaslambda2=zeros(length(pgamma),Kbeta);
    absbiaslambda0=zeros(length(pgamma));
    absbiaslambda1=zeros(length(pgamma),Kalpha);
    absbiaslambda2=zeros(length(pgamma),Kbeta);
    relative_biaslambda0=zeros(length(pgamma));
    relative_biaslambda1=zeros(length(pgamma),Kalpha);
    relative_biaslambda2=zeros(length(pgamma),Kbeta);


    initial_lambda0=zeros(27,1);
    initial_lambda1=zeros(27,3);
    initial_lambda1(lambda1_indi==1)=2;
    initial_lambda2=zeros(27,5);
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
                temp=DFSM_likelihood(response(i,:),qmatrix,truekey,L,initial_lambda0,initial_lambda1,initial_lambda2,trueprofile,Kalpha,Kbeta);
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
            for j=1:L
                qj=qmatrix(((j-1)*4+1):(4*j), :);
                plength=J1(j)+J2(j)+1;
                initial=[0,ones(1, plength-1)]; % the initial values could be more informative later
                A=[];
                b=[];
                Aeq=[];
                beq=[];
                lb=[-2,zeros(1,plength-1)];
                ub=[2,4.*ones(1,plength-1)];
                output=fmincon(@(temp_para)DFSM_itemnlikelihood_unknown(response(:,j),posterior,qj,truekey(j),N,lambda1_indi(j,:),lambda2_indi(j,:),J1(j),J2(j),temp_para,K,trueprofile),initial,A,b,Aeq,beq,lb,ub);

                est_lambda0(j)=output(1);
                %temp=find(lambda1_indi(j,:)==1); % find locations of non-zero lambda1 entries
                est_lambda1(j,(lambda1_indi(j,:)==1))=output(2:(1+J1(j)));
                %temp=find(lambda2_indi(j,:)==1); % find locations of non-zero lambda1 entries
                est_lambda2(j,(lambda2_indi(j,:)==1))=output((2+J1(j)):plength);
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
                temp=DFSM_likelihood(response(i,:),qmatrix,truekey,L,initial_lambda0,initial_lambda1,initial_lambda2,trueprofile,Kalpha,Kbeta);
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
            for j=1:L
                qj=qmatrix(((j-1)*4+1):(4*j), :);
                plength=J1(j)+J2(j)+1;
                initial=[0,ones(1, plength-1)]; % the initial values could be more informative later
                A=[];
                b=[];
                Aeq=[];
                beq=[];
                lb=[-2,zeros(1,plength-1)];
                ub=[2,4.*ones(1,plength-1)];
                output=fmincon(@(temp_para)DFSM_itemnlikelihood_unknown(response(:,j),posterior,qj,truekey(j),N,lambda1_indi(j,:),lambda2_indi(j,:),J1(j),J2(j),temp_para,K,trueprofile),initial,A,b,Aeq,beq,lb,ub);

                est_lambda0(j)=output(1);
                %temp=find(lambda1_indi(j,:)==1); % find locations of non-zero lambda1 entries
                est_lambda1(j,(lambda1_indi(j,:)==1))=output(2:(1+J1(j)));
                %temp=find(lambda2_indi(j,:)==1); % find locations of non-zero lambda1 entries
                est_lambda2(j,(lambda2_indi(j,:)==1))=output((2+J1(j)):plength);
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

        % check item parameter recovery
        biaslambda0(penalty)=mean(est_lambda0-lambda0);
        relative_biaslambda0(penalty)=mean((est_lambda0-lambda0)./lambda0);
        absbiaslambda0(penalty)=mean(abs(est_lambda0-lambda0));
        biaslambda1(penalty,:)=mean(est_lambda1-lambda1).*(27/15);
        biaslambda2(penalty,:)=mean(est_lambda2-lambda2).*27./(sum(lambda2_indi));
        absbiaslambda1(penalty,:)=mean(abs(est_lambda1-lambda1)).*(27/15);
        absbiaslambda2(penalty,:)=mean(abs(est_lambda2-lambda2)).*27./(sum(lambda2_indi));


        relative_biaslambda1(penalty,:)=mean((est_lambda1-lambda1)./(lambda1+0.00001)).*(27/15);
        relative_biaslambda2(penalty,:)=mean((est_lambda2-lambda2)./(lambda2+0.00001)).*27./(sum(lambda2_indi));

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
        true_sparsity=(prior==0)';
        ebic(penalty,3)=sum(abs(est_sparsity-true_sparsity))/(2^K); %error rate of the sparsity
        ebic(penalty,4)=mean(abs(pi_final'-prior)); % mean absolute bias
        % proportion of true 0's that are correctly recovered
        est_sparsity=find(pi_final==0);
        true_sparsity=find(prior==0);
        ebic(penalty,5)=length(intersect(est_sparsity, true_sparsity))/sum(prior==0);
        % proportion of true non-zero's that are correctly recovered % TPR in
        % Gu & Xu (2019)
        est_sparsity1=find(pi_final>0);
        true_sparsity1=find(prior>0);
        ebic(penalty,6)=length(intersect(est_sparsity1, true_sparsity1))/sum(prior>0);
        % 1-FDR: proportion of selected patterns that are true patterns
        ebic(penalty,7)=length(intersect(est_sparsity1, true_sparsity1))/sum(pi_final>0);
        ebic(penalty,8)=marginal_log; % log-marginal likelihood, no penalty



        filename1 = ['DFSM_full_moderatecorr_N2000_' int2str(rep) '.mat'];
        save(filename1)
    end


    % Since we did not save all item parameters, we need to re-run EM again to
    % obtain final item parameters and to estimate facet and profile recovery
    % Use the best solution
    %% Re-run EM to obtain optimal item parameters so that we can compute mastery profile
    % In previous run, we don't save all item parameters per tuning parameter
    % but our solution path is useful to definie pi_final_p

    % After deciding on the sparsity structure of tau12, re-run EM to
    % obtain EBIC
    pi_initial=1/(2^K).*ones(1,2^K);
    bestsolution=find(ebic(:,1)==min(ebic(:,1))); % based on BIC
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
            temp=DFSM_likelihood(response(i,:),qmatrix,truekey,L,initial_lambda0,initial_lambda1,initial_lambda2,trueprofile,Kalpha,Kbeta);
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
        for j=1:L
            qj=qmatrix(((j-1)*4+1):(4*j), :);
            plength=J1(j)+J2(j)+1;
            initial=[0,ones(1, plength-1)]; % the initial values could be more informative later
            A=[];
            b=[];
            Aeq=[];
            beq=[];
            lb=[-2,zeros(1,plength-1)];
            ub=[2,4.*ones(1,plength-1)];
            output=fmincon(@(temp_para)DFSM_itemnlikelihood_unknown(response(:,j),posterior,qj,truekey(j),N,lambda1_indi(j,:),lambda2_indi(j,:),J1(j),J2(j),temp_para,K,trueprofile),initial,A,b,Aeq,beq,lb,ub);

            est_lambda0(j)=output(1);
            %temp=find(lambda1_indi(j,:)==1); % find locations of non-zero lambda1 entries
            est_lambda1(j,(lambda1_indi(j,:)==1))=output(2:(1+J1(j)));
            %temp=find(lambda2_indi(j,:)==1); % find locations of non-zero lambda1 entries
            est_lambda2(j,(lambda2_indi(j,:)==1))=output((2+J1(j)):plength);
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


    %%
    estimated_profile=zeros(N,Kalpha+Kbeta);
    est_profileprob=zeros(N,2^(Kalpha+Kbeta));
    equ_prior=ones(256,1).*(1/256);

    estimated_profile_equprior=zeros(N,Kalpha+Kbeta);
    est_profileprob_equprior=zeros(N,2^(Kalpha+Kbeta));
    for i=1:N  % let's focus on the first 1000 people as the algorithm can be slow at the moment
        [estprofile, profile_prob]=DFSM_MAP(response(i,:),qmatrix,truekey,L,est_lambda0,est_lambda1,est_lambda2,pi_final',trueprofile,Kalpha, Kbeta);
        estimated_profile(i,:)=estprofile;
        est_profileprob(i,:)=profile_prob';
    end

    count=0;
    for i=1:N
        if (sum(abs(estimated_profile(i,:)-truep(i,:)))==0)
            count=count+1;
        end
    end
    facetprofilerecovery=count/N;


    K=Kalpha+Kbeta;
    attribute_recovery=zeros(K,1);
    for k=1:K
        attribute_recovery(k)=sum(estimated_profile(1:N,k)==truep(1:N,k))/N;
    end
    facetrecovery=mean(attribute_recovery);


    filename1 = ['DFSM_full_moderatecorr_N2000_' int2str(rep) '.mat'];
    save(filename1)
    [aa,~,c] = unique(estimated_profile,'rows');
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
            if (length(observed_count)==num_response(j) & min(observed_count.*temprow(g,9))>=5)
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
    filename1 = ['DFSM_full_moderatecorr_N2000_' int2str(rep) '.mat'];
    save(filename1)


end