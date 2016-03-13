function [labels,weight,theta] = lda_gibbs_sampler(documents,K,V)
% gibbs_sampler inference for LDA 
% based on the method in "Parameter estimation for text analysis. Gregor 
% Heinrich. Technical Report. "
%-------------------------------------------------------------
% author: Hanchen Xiong   University of Innsbruck
%-------------------------------------------------------------

% the number of documents in copora 
M=size(documents,1);
% the number of words in each document
N_m=zeros(M,1);
for m=1:M
    N_m=size(documents{m,1},2);
end;
% the label of words in each document
labels=documents;

% output data
weight=zeros(M,K);         % p(topic|m)    |topic|=K, |document|=M
theta=zeros(K,V);          % p(word|topic) |word|=V, |topic|=K

% hyperparameters control 
alpha=ones(1,K);           % all initialized wiht symmetric paprameters
beta=0.05*ones(1,V);



% threes data structure to recorde the count
N_W_Z=zeros(K,V);  % the count of words (W) which are assigend to topics (Z)
N_Z_D=zeros(M,K);  % the count of topics (Z) which are assigend to documents (D)
N_Z=zeros(K);      % the sum of all count \sum_W{N_W_Z} for each Z

% --------------------for visualization----------------------------------
w_position=zeros(25,2);    % the 2D position correspond different indices
%                          % specify word index-position correspondence
        % -------------------------------------------------------------
        % |  1  |  2  |  3  |  4  |  5  | 
        % |  6  |  7  |  8  |  9  |  10 |
        % |  11 |  12 |  13 |  14 |  15 |
        % |  16 |  17 |  18 |  19 |  20 |
        % |  21 |  22 |  23 |  24 |  25 |
        %---------------------------------------------------------------
for i=1:5
    for j=1:5
        w_position((i-1)*5+j,1)=i;
        w_position((i-1)*5+j,2)=j;
    end;
end;


% initialization
for m=1:M
    for n=1:N_m
        w=documents{m,1}(n);                % the index w of word W_{m,n}
        k=find(mnrnd(1,ones(1,K)/K));       % sample a topic associated with word w;
        labels{m,1}(n)=k;                   % label with the selected topic k; 
        N_Z_D(m,k)=N_Z_D(m,k)+1;   
        N_W_Z(k,w)=N_W_Z(k,w)+1;   
        N_Z(k)=N_Z(k)+1;
    end;
end;
% gibbs sampling until converge
cstep=0;
nstep=100;

while cstep<nstep
    cstep=cstep+1
%     figure;
%     set (gcf,'Position',[400,400,K*100,80], 'color','w')
    for m=1:M
        for n=1:N_m
            w=documents{m,1}(n);       % the index w of word W_{m,n}
            k=labels{m,1}(n);          % the corresponding label (index of topic)
            
            % all count realated word m and topic k are decreased by 1
            N_Z_D(m,k)=N_Z_D(m,k)-1;   
            N_W_Z(k,w)=N_W_Z(k,w)-1;   
            N_Z(k)=N_Z(k)-1;
              
            % sampling a new topic label for word W_{m,n}
            sample_prob=zeros(1,K);  % sampling probability
            for z=1:K
                sum_z_v=sum(N_W_Z(z,:))+sum(beta);
                sample_prob(z)=((N_W_Z(z,w)+1)/sum_z_v)*(N_Z_D(m,z)+alpha(z));
            end;
            
            for z=1:K
                norm_sample_prob=sample_prob/sum(sample_prob);
            end;
            
            new_l=find(mnrnd(1,norm_sample_prob)); % new label
            
            % updat new label and corresponding counts
            labels{m,1}(n)=new_l;
            N_Z_D(m,new_l)=N_Z_D(m,new_l)+1;   
            N_W_Z(new_l,w)=N_W_Z(new_l,w)+1;   
            N_Z(new_l)=N_Z(new_l)+1;
        end;
    end;
          
    % visualize parameter theta   
    if ismember(cstep,[1,10,30,80,150,300,500])
        index_row=find([1,10,30,80,150,300,500]==cstep);
        images=cell(K,1);
        for kk=1:K
            images{kk,1}=zeros(5,5);
            sum_k_v=sum(N_W_Z(kk,:))+sum(beta);
            for v=1:V
                theta(kk,v)=(N_W_Z(kk,v)+1)/sum_k_v;
                v_pos_row=w_position(v,1);
                v_pos_col=w_position(v,2);
                images{kk,1}(v_pos_row,v_pos_col)=theta(kk,v);
            end;
            subplot(7,K,(index_row-1)*10+kk);
            imagesc(images{kk,1});
            colormap('jet');
            axis off;
        end;
    end;
end;
% read out parameters 
for kk=1:K  
    sum_k_v=sum(N_W_Z(kk,:))+sum(beta);
    for v=1:V
        theta(kk,v)=(N_W_Z(kk,v)+1)/sum_k_v;
    end;
end;
end

