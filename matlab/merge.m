n = 100000;
k = 330;
p = 0.01;
beta = 0.05;
binomial = makedist('Binomial',k,p);
alpha = icdf(binomial,(n-k)/n);
truncated = truncate(binomial,alpha,inf);
A_stimulus = random(truncated, 1, k);
A_stimulus = A_stimulus*(1+beta);
B_stimulus = random(truncated, 1, k);
B_stimulus = B_stimulus*(1+beta);

bernoulli = makedist('Binomial',1,p);
A_connectome = random(bernoulli,k,k);
B_connectome = random(bernoulli,k,k);
A_winners = ones(1, k);
B_winners = ones(1,k);

binomial2 = makedist('Binomial',2*k,p);
alpha2 = icdf(binomial2, (n-k)/n);
truncated2 = truncate(binomial2,alpha2,inf);
C_inputs = random(truncated2, 1, k);
C_winners = ones(1,k);
A_C_connectome = zeros(k, k);
B_C_connectome = zeros(k, k);
for i = 1:k
    C_input = random(truncated2);
    divided_input = randsample(2*k,C_input);
    for j = 1:length(divided_input)
        elm = divided_input(j);
        if (elm <= k) % Let this mean it came from A
            A_C_connectome(elm,i) = (1+beta);
        else
            B_C_connectome(elm-k,i) = (1+beta);
        end
    end
end
C_connectome = random(bernoulli,k,k);

T=30;
A_new_winner_at_t = zeros(1,T);
A_size_winners_at_t = zeros(1,T);
B_new_winner_at_t = zeros(1,T);
B_size_winners_at_t = zeros(1,T);
C_new_winner_at_t = zeros(1,T);
C_size_winners_at_t = zeros(1,T);

% Going in at each stage we have:
% winners: binary array of the w neurons we have memorized. 1 if neuron won
% in previous stage, 0 otherwise.
% connectome: w x w matrix of synapses between neurons in winners.
% stimulus: array of total input weight from stimulus to each winner.

A_winner_indices = find(A_winners == 1);
A_size = size(A_connectome,1);

binomial3 = makedist('Binomial', 3*k,p);  % for sampling new winners in C
alpha3 = icdf(binomial3, (n-k)/n);  % should it be (n-k-w)/n
truncated3 = truncate(binomial3,alpha3,inf);

for t=1:T
    % work on A
    % inputs: A_stimulus
    A_w = size(A_connectome,1);
    A_newcandidates = random(truncated2,1,k);
    A_supportinputs = zeros(1,A_w);
    for i = 1:A_w
        A_supportinputs(i) = A_stimulus(i);
        for j = 1:A_w
            A_supportinputs(i) = A_supportinputs(i) + A_connectome(j,i)*A_winners(j);
        end
    end
    
    both = [A_supportinputs A_newcandidates];
    [~,I] = maxk(both,k);
    A_num_new_winners = nnz(I > A_w);
    new_A_w = A_w+A_num_new_winners;
    new_A_winners = zeros(1,new_A_w);
    A_new_winner_inputs = zeros(1,A_num_new_winners);
    j=1;
    for i = I
        if i <= A_w
            new_A_winners(i) = 1;
        else
            new_A_winners(A_w+j) = 1;
            A_new_winner_inputs(j) = both(i);
            j = j+1;
        end
    end
    
    % can construct A_new_stimulus, expands by A_num_new_winners
    % note that this won't set A_new_stimulus values for new winners
    new_A_stimulus = zeros(1,new_A_w);
    for i = 1:A_w
        new_A_stimulus(i) = A_stimulus(i);
        if new_A_winners(i) == 1
            new_A_stimulus(i) = new_A_stimulus(i)*(1+beta);
        end
    end
    
    % for each new winner, sample how much came from stim, recurrent, B
    % the three columns are total amount from STIM, RECURRENT (A), B
    A_recurrent_inputs = zeros(A_num_new_winners,2);
    for i=1:A_num_new_winners
        r = randsample(2*k, A_new_winner_inputs(i));
        stim_input = nnz(k >= r);
        new_A_stimulus(A_w+i) = stim_input*(1+beta);
        A_recurrent_inputs(i) = nnz(r > k);
    end

    % can construct A_new_connectome, expands by A_num_new_winners
    % later: fill in inputs from B to new A
    new_A_connectome = zeros(new_A_w,new_A_w);
    for i=1:A_w
        for j=1:A_w
            new_A_connectome(j,i) = A_connectome(j,i);
            if new_A_winners(i) == 1 && A_winners(j) == 1
                new_A_connectome(j,i) = new_A_connectome(j,i)*(1+beta);
            end
        end
        for j = (A_w+1):new_A_w
            % Whether the new neuron j connects to neuron i
            new_A_connectome(j,i) = random(bernoulli);
        end
    end
    A_previous_winner_indices = find(A_winners == 1);
    for i=1:A_num_new_winners
        % Working on column w+i (inputs into that neuron)
        recurrent_input = A_recurrent_inputs(i);
        inputs = randsample(A_previous_winner_indices, recurrent_input);
        for j=inputs
            new_A_connectome(j,A_w+i) = 1+beta;
        end
        for j = (A_w+1):new_A_w
            new_A_connectome(j,A_w+i) = random(bernoulli);
        end
        for j = 1:A_w
            if (A_winners(j) == 0)
                new_A_connectome(j,A_w+i) = random(bernoulli);
            end
        end
    end
    
    % can construct new_A_C_connectome
    C_w = size(C_connectome,1);
    new_A_C_connectome = zeros(new_A_w, C_w);
    for i = 1:A_w
        for j = 1:C_w
            new_A_C_connectome(i,j) = A_C_connectome(i,j);
        end
    end
    for i = (A_w+1):new_A_w
        for j = 1:C_w
            new_A_C_connectome(i,j) = random(bernoulli);
        end
    end
    
    A_stimulus = new_A_stimulus;
    A_connectome = new_A_connectome;
    A_winners = new_A_winners;
    A_C_connectome = new_A_C_connectome;
    A_new_winner_at_t(t) = A_num_new_winners;
    A_size_winners_at_t(t) = new_A_w;
    
    % work on B
    % inputs: B_stimulus
    B_w = size(B_connectome,1);
    B_newcandidates = random(truncated2,1,k);
    B_supportinputs = zeros(1,B_w);
    for i = 1:B_w
        B_supportinputs(i) = B_stimulus(i);
        for j = 1:B_w
            B_supportinputs(i) = B_supportinputs(i) + B_connectome(j,i)*B_winners(j);
        end
    end
    
    both = [B_supportinputs B_newcandidates];
    [~,I] = maxk(both,k);
    B_num_new_winners = nnz(I > B_w);
    new_B_w = B_w+B_num_new_winners;
    new_B_winners = zeros(1,new_B_w);
    B_new_winner_inputs = zeros(1,B_num_new_winners);
    j=1;
    for i = I
        if i <= B_w
            new_B_winners(i) = 1;
        else
            new_B_winners(B_w+j) = 1;
            B_new_winner_inputs(j) = both(i);
            j = j+1;
        end
    end
    
    % can construct B_new_stimulus, expands by B_num_new_winners
    % note that this won't set B_new_stimulus values for new winners
    new_B_stimulus = zeros(1,new_B_w);
    for i = 1:B_w
        new_B_stimulus(i) = B_stimulus(i);
        if new_B_winners(i) == 1
            new_B_stimulus(i) = new_B_stimulus(i)*(1+beta);
        end
    end
    
    % for each new winner, sample how much came from stim, recurrent, B
    % the three columns are total amount from STIM, RECURRENT (A), B
    B_recurrent_inputs = zeros(B_num_new_winners,2);
    for i=1:B_num_new_winners
        r = randsample(2*k, B_new_winner_inputs(i));
        stim_input = nnz(k >= r);
        new_B_stimulus(B_w+i) = stim_input*(1+beta);
        B_recurrent_inputs(i) = nnz(r > k);
    end

    % can construct B_new_connectome, expands by B_num_new_winners
    new_B_connectome = zeros(new_B_w,new_B_w);
    for i=1:B_w
        for j=1:B_w
            new_B_connectome(j,i) = B_connectome(j,i);
            if new_B_winners(i) == 1 && B_winners(j) == 1
                new_B_connectome(j,i) = new_B_connectome(j,i)*(1+beta);
            end
        end
        for j = (B_w+1):new_B_w
            % Whether the new neuron j connects to neuron i
            new_B_connectome(j,i) = random(bernoulli);
        end
    end
    B_previous_winner_indices = find(B_winners == 1);
    for i=1:B_num_new_winners
        % Working on column w+i (inputs into that neuron)
        recurrent_input = B_recurrent_inputs(i);
        inputs = randsample(B_previous_winner_indices, recurrent_input);
        for j=inputs
            new_B_connectome(j,B_w+i) = 1+beta;
        end
        for j = (B_w+1):new_B_w
            new_B_connectome(j,B_w+i) = random(bernoulli);
        end
        for j = 1:B_w
            if (B_winners(j) == 0)
                new_B_connectome(j,B_w+i) = random(bernoulli);
            end
        end
    end
    
    % can construct new_B_C_connectome
    new_B_C_connectome = zeros(new_B_w, C_w);
    for i = 1:B_w
        for j = 1:C_w
            new_B_C_connectome(i,j) = B_C_connectome(i,j);
        end
    end
    for i = (B_w+1):new_B_w
        for j = 1:C_w
            new_B_C_connectome(i,j) = random(bernoulli);
        end
    end
    
    B_stimulus = new_B_stimulus;
    B_connectome = new_B_connectome;
    B_winners = new_B_winners;
    B_C_connectome = new_B_C_connectome;
    B_new_winner_at_t(t) = B_num_new_winners;
    B_size_winners_at_t(t) = new_B_w;
    
    A_w = new_A_w;
    B_w = new_B_w;
    
    % now we can work on C
    C_w = size(C_connectome,1);
    C_newcandidates = random(truncated3,1,k);
    C_supportinputs = zeros(1,C_w);
    for i = 1:C_w
        for j = 1:A_w
            C_supportinputs(i) = C_supportinputs(i) + A_C_connectome(j,i)*A_winners(j);
        end
        for j = 1:B_w
            C_supportinputs(i) = C_supportinputs(i) + B_C_connectome(j,i)*B_winners(j);
        end
        for j = 1:C_w
            C_supportinputs(i) = C_supportinputs(i) + C_connectome(j,i)*C_winners(j);
        end
    end
    both = [C_supportinputs C_newcandidates];
    [~,I] = maxk(both,k);
    C_num_new_winners = nnz(I > C_w);
    new_C_w = C_w+C_num_new_winners;
    new_C_winners = zeros(1,new_C_w);
    C_new_winner_inputs = zeros(1,C_num_new_winners);
    j=1;
    for i = I
        if i <= C_w
            new_C_winners(i) = 1;
        else
            new_C_winners(C_w+j) = 1;
            C_new_winner_inputs(j) = both(i);
            j = j+1;
        end
    end
    
    % for each new winner, sample how much came from A, B, recurrent
    % the three columns are total amount from A, B, recurrent (C)
    C_new_winner_input_sources = zeros(C_num_new_winners,3);
    for i=1:C_num_new_winners
        r = randsample(3*k, C_new_winner_inputs(i));
        C_new_winner_input_sources(i,1) = nnz(k >= r);
        C_new_winner_input_sources(i,2) = nnz(2*k >= r & r > k);
        C_new_winner_input_sources(i,3) = nnz(r > 2*k);
    end
    
    % can construct C_new_connectome, expands by C_num_new_winners
    new_C_connectome = zeros(new_C_w,new_C_w);
    for i=1:C_w
        for j=1:C_w
            new_C_connectome(j,i) = C_connectome(j,i);
            if new_C_winners(i) == 1 && C_winners(j) == 1
                new_C_connectome(j,i) = new_C_connectome(j,i)*(1+beta);
            end
        end
        for j = (C_w+1):new_C_w
            % Whether the new neuron j connects to neuron i
            new_C_connectome(j,i) = random(bernoulli);
        end
    end
    C_previous_winner_indices = find(C_winners == 1);
    for i=1:C_num_new_winners
        % Working on column w+i (inputs into that neuron)
        recurrent_input = C_new_winner_input_sources(i,3);
        inputs = randsample(C_previous_winner_indices, recurrent_input);
        for j=inputs
            new_C_connectome(j,C_w+i) = 1+beta;
        end
        for j = (C_w+1):new_C_w
            new_C_connectome(j,C_w+i) = random(bernoulli);
        end
        for j = 1:C_w
            if (C_winners(j) == 0)
                new_C_connectome(j,C_w+i) = random(bernoulli);
            end
        end
    end
    
    % Now work on new_A_C_connectome
    % expands by C_num_new_winners columns
    % for new columns, sample input based on
    % C_new_winner_input_sources(i,1)
    % can construct new_A_C_connectome
    new_A_C_connectome = zeros(A_w, new_C_w);
    for i = 1:A_w
        for j = 1:C_w
            new_A_C_connectome(i,j) = A_C_connectome(i,j);
        end
        if A_winners(i) == 1 && new_C_winners(j) == 1
            new_A_C_connectome(i,j) = new_A_C_connectome(i,j)*(1+beta);
        end
    end
    A_winner_indices = find(A_winners == 1);
    for j = 1:C_num_new_winners
        % Working on column C_w+j (inputs into that neuron)
        from_A = C_new_winner_input_sources(j,1);
        inputs = randsample(A_winner_indices, from_A);
        for i=inputs
            new_A_C_connectome(i,C_w+j) = 1+beta;
        end
        for i = 1:A_w
            if (A_winners(i) == 0)
                new_A_C_connectome(i,C_w+j) = random(bernoulli);
            end
        end
    end
    
    new_B_C_connectome = zeros(B_w, new_B_w);
    for i = 1:B_w
        for j = 1:C_w
            new_B_C_connectome(i,j) = B_C_connectome(i,j);
        end
        if B_winners(i) == 1 && new_C_winners(j) == 1
            new_B_C_connectome(i,j) = new_B_C_connectome(i,j)*(1+beta);
        end
    end
    B_winner_indices = find(B_winners == 1);
    for j = 1:C_num_new_winners
        % Working on column C_w+j (inputs into that neuron)
        from_B = C_new_winner_input_sources(j,2);
        inputs = randsample(B_winner_indices, from_B);
        for i=inputs
            new_B_C_connectome(i,C_w+j) = 1+beta;
        end
        for i = 1:B_w
            if (B_winners(i) == 0)
                new_B_C_connectome(i,C_w+j) = random(bernoulli);
            end
        end
    end
    
    C_connectome = new_C_connectome;
    A_C_connectome = new_A_C_connectome;
    B_C_connectome = new_B_C_connectome;
    C_winners = new_C_winners;
    C_new_winner_at_t(t) = C_num_new_winners;
    C_size_winners_at_t(t) = new_C_w;
end

       

