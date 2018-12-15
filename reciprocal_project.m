n = 100000;
k = 316;
p = 0.01;
beta = 0.08;
normal = makedist('Binomial',k,p);
alpha = icdf(normal,(n-k)/n);
truncated = truncate(normal,alpha,inf);
stimulus = random(truncated, 1, k);
stimulus = stimulus*(1+beta);
bernoulli = makedist('Binomial',1,p);
A_connectome = random(bernoulli,k,k);
winners = ones(1, k);

T=41;
A_new_winner_at_t = zeros(1,T);
A_size_winners_at_t = zeros(1,T);
B_new_winner_at_t = zeros(1,T);
B_size_winners_at_t = zeros(1,T);

% Going in at each stage we have:
% winners: binary array of the w neurons we have memorized. 1 if neuron won
% in previous stage, 0 otherwise.
% connectome: w x w matrix of synapses between neurons in winners.
% stimulus: array of total input weight from stimulus to each winner.
normal2 = makedist('Binomial', 2*k,p);
alpha2 = icdf(normal2, (n-k)/n);  % should it be (n-k-w)/n
truncated2 = truncate(normal2,alpha2,inf);
for t=1:1
    w = size(winners,2);
    newcandidates = random(truncated2,1,k);
    supportinputs = zeros(1,w);
    for i = 1:w
        supportinputs(i) = stimulus(i);
        for j = 1:w
            supportinputs(i) = supportinputs(i) + A_connectome(j,i)*winners(j);
        end
    end
    both = [supportinputs newcandidates];
    [B,I] = maxk(both,k);
    num_new_winners = nnz(I > w);
    new_w = w+num_new_winners;
    new_winners = zeros(1,new_w);
    new_winner_inputs = zeros(1,num_new_winners);
    j=1;
    for i = I
        if i <= w
            new_winners(i) = 1;
        else
            new_winners(w+j) = 1;
            new_winner_inputs(j) = both(i);
            j = j+1;
        end
    end
    % Work on expanding the connectome
    % Need to figure out input from stimulus vs old assembly for new winners
    % If X, Y both distribution by normal(mu, sigma) and X+Y = b
    % Then distribution of X (and Y) is Normal((1/2)b, sigma/sqrt(2))

    % Update the stimulus vector
    % instead of allocaitng new array here, could append to old one
    new_stimulus = zeros(1,new_w);
    for i = 1:w
        new_stimulus(i) = stimulus(i);
        if new_winners(i) == 1
            new_stimulus(i) = new_stimulus(i)*(1+beta);
        end
    end
    recurrent_inputs = zeros(1,num_new_winners);
    for i = 1:num_new_winners
        b = new_winner_inputs(i);
        % Randomly generate how much came from the previous assembly
        % Input from stimulus or previous assembly equally likely, 
        % randomly choose b out of 2k.
        divided_input = randsample(2*k,b);
        recurrent_input = nnz(divided_input <= k);
        recurrent_inputs(i) = recurrent_input;
        new_stimulus(w+i) = (b-recurrent_input)*(1+beta);
    end
    % Have new winner array, new stimulus array
    % Last thing to do is update the connectome
    % Is there a more efficient way to reallocate?
    new_connectome = zeros(new_w,new_w);
    for i=1:w
        for j=1:w
            new_connectome(j,i) = A_connectome(j,i);
            if new_winners(i) == 1 && winners(j) == 1
                new_connectome(j,i) = new_connectome(j,i)*(1+beta);
            end
        end
        for j = (w+1):new_w
            % Whether the new neuron j connects to neuron i
            new_connectome(j,i) = random(bernoulli);
        end
    end
    previous_winner_indices = find(winners == 1);
    for i=1:num_new_winners
        % Working on column w+i (inputs into that neuron)
        recurrent_input = recurrent_inputs(i);
        inputs = randsample(previous_winner_indices, recurrent_input);
        for j=inputs
            new_connectome(j,w+i) = 1+beta;
        end
        for j = (w+1):new_w
            new_connectome(j,w+i) = random(bernoulli);
        end
        for j=1:w
            if (winners(j) == 0)
                new_connectome(j,w+i) = random(bernoulli);
            end
        end
    end
    A_new_winner_at_t(t) = num_new_winners;
    A_size_winners_at_t(t) = size(new_winners,2);
    winners = new_winners;
    stimulus = new_stimulus;
    A_connectome = new_connectome;
end

A_winner_indices = find(winners == 1);
input_B = random(truncated, 1, k);
A_size = size(A_connectome,1);
A_B_connectome = zeros(A_size, k);
for i=1:k % for each winner in B
    inputs_to_neuron = randsample(k,input_B(i));
    for j=inputs_to_neuron
        A_B_connectome(j,i) = 1+beta;
    end
    for j=(k+1):A_size
        A_B_connectome(j,i) = random(bernoulli);
    end
end
B_connectome = random(bernoulli,k,k);
B_A_connectome = random(bernoulli,k,A_size);
B_winners = ones(1,k);
A_winners = winners;
A_stimulus = stimulus;

B_size_winners_at_t(t) = k;

normal3 = makedist('Binomial', 3*k,p);
alpha3 = icdf(normal3, (n-k)/n);  % should it be (n-k-w)/n
truncated3 = truncate(normal3,alpha3,inf);

for t=2:T
    % work on A
    % inputs: stimulus, recurrent, and from B
    
    % calculate inputs to A_connectome neurons (called A_supportinputs)
    % sum of A_stimulus, recurrent (from A_connectome), 
    % and input from B (from B_A_connectome)
    A_w = size(A_winners,2);
    B_w = size(B_winners,2);
    A_supportinputs = zeros(1,A_w);
    for i = 1:A_w
        A_supportinputs(i) = A_stimulus(i);
        for j = 1:A_w
            A_supportinputs(i) = A_supportinputs(i) + A_connectome(j,i)*A_winners(j);
        end
        for j = 1:B_w
            A_supportinputs(i) = A_supportinputs(i) + B_A_connectome(j,i)*B_winners(j);
        end
    end
    
    % generate potential new winners (called A_newcandidates)
    % sample from outside A_connectome from normal(3k,sqrt(3)*sigma)
    A_newcandidates = random(truncated3,1,k);
    
    % both = [A_supportinputs, A_newcandidates]
    % [B,I] = maxk(both,k);  % cap operation
    % generate A_new_winners based of I, and A_num_new_winners
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
    new_A_stimulus = zeros(1,new_A_w);
    for i = 1:A_w
        new_A_stimulus(i) = A_stimulus(i);
        if new_A_winners(i) == 1
            new_A_stimulus(i) = new_A_stimulus(i)*(1+beta);
        end
    end
    
    % for each new winner, sample how much came from stim, recurrent, B
    % the three columns are total amount from STIM, RECURRENT (A), B
    A_new_winner_input_sources = zeros(A_num_new_winners,3);
    for i=1:A_num_new_winners
        r = randsample(3*k, A_new_winner_inputs(i));
        A_new_winner_input_sources(i,1) = nnz(k >= r);
        new_A_stimulus(A_w+i) = A_new_winner_input_sources(i,1)*(1+beta);
        A_new_winner_input_sources(i,2) = nnz(2*k >= r & r > k);
        A_new_winner_input_sources(i,3) = nnz(r > 2*k);
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
        recurrent_input = A_new_winner_input_sources(i,2);
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
    
    A_stimulus = new_A_stimulus;
    A_connectome = new_A_connectome;
    
    % work on B
    % calculate inputs to B_connectome neurons (called B_supportinputs)
    % sum of A_B inputs (from A_B_connectome), and recurrent from B
    B_supportinputs = zeros(1,B_w);
    for i = 1:B_w
        for j = 1:A_w
            B_supportinputs(i) = B_supportinputs(i) + A_B_connectome(j,i)*A_winners(j);
        end
        for j = 1:B_w
            B_supportinputs(i) = B_supportinputs(i) + B_connectome(j,i)*B_winners(j);
        end
    end
    
    % generate potential new winners (called B_newcandidates)
    % sample from outside B_connectome from normal2 
    B_newcandidates = random(truncated2,1,k);
    
    % both = [B_supportinputs, B_newcandidates]
    % [B,I] = maxk(both,k);  % cap operation
    % generate B_new_winners based of I, and B_num_new_winners
    both = [B_supportinputs B_newcandidates];
    [B,I] = maxk(both,k);
    
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
    
    % for each new winner, sample how much came from A or recurrent
    % order is A, RECURRENT
    B_new_winner_input_sources = zeros(A_num_new_winners,2);
    for i=1:B_num_new_winners
        r = randsample(2*k, B_new_winner_inputs(i));
        from_A = nnz(k >= r);
        B_new_winner_input_sources(i,1) = from_A;
        B_new_winner_input_sources(i,2) = B_new_winner_inputs(i) - from_A;
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
        recurrent_input = B_new_winner_input_sources(i,2);
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
    
    B_connectome = new_B_connectome;
    
    % construct new_A_B_connectome
    % expands by A_num_new_winners rows, B_num_new_winners columns
    % for each row a, if a was (previous round) A winner and column b
    %  won now, then multiply by (1+beta)
    % for i=1,B_num_new_winners (columns), randomly sample which rows
    % also expands rows by A_num_new_winners, connections are bernoulli
    new_A_B_connectome = zeros(new_A_w,new_B_w);
    for a=1:A_w
        for b=1:B_w
            new_A_B_connectome(a,b) = A_B_connectome(a,b);
            if new_B_winners(b) == 1 && A_winners(a) == 1
                new_A_B_connectome(a,b) = new_A_B_connectome(a,b)*(1+beta);
            end 
        end
    end
    for b=1:B_num_new_winners
        input_from_A = B_new_winner_input_sources(b,1);
        inputs = randsample(A_previous_winner_indices, input_from_A);
        for j=inputs
            new_A_B_connectome(j,B_w+b) = 1+beta;
        end
        for a=1:A_w
            if (A_winners(a) == 0)
                new_A_B_connectome(a,B_w+b) = random(bernoulli);
            end
        end
    end
    for a=(A_w+1):new_A_w
        for b=1:new_B_w
            new_A_B_connectome(a,b) = random(bernoulli);
        end
    end
    A_B_connectome = new_A_B_connectome;
    
    % construct B_A_new_connectome
    % expands by B_num_new_winners rows, A_num_new_winners columns
    % for each row i, if i was (previous round B winner and column j
    % won now, then multiply by (1+beta)
    % also for i=1,A_num_new_winners (columns), randomly sample which rows
    % also expands rows by B_num_new_winners, connections bernoulli
    new_B_A_connectome = zeros(new_B_w,new_A_w);
    for b=1:B_w
        for a=1:A_w
            new_B_A_connectome(b,a) = B_A_connectome(b,a);
            if new_A_winners(a) == 1 && B_winners(b) == 1
                new_B_A_connectome(b,a) = new_B_A_connectome(b,a)*(1+beta);
            end
        end
    end
    for a=1:A_num_new_winners
        input_from_B = A_new_winner_input_sources(a,3);
        inputs = randsample(B_previous_winner_indices, input_from_B);
        for j=inputs
            new_B_A_connectome(j,A_w+a) = 1+beta;
        end
        for b=1:B_w
            if (B_winners(b) == 0)
                new_B_A_connectome(b,A_w+a) = random(bernoulli);
            end
        end
    end
    for b=(B_w+1):new_B_w
        for a=1:new_A_w
            new_B_A_connectome(b,a) = random(bernoulli);
        end
    end
    % make sure everything set to new versions
    B_A_connectome = new_B_A_connectome;
    A_winners = new_A_winners;
    B_winners = new_B_winners;
    
    A_new_winner_at_t(t) = A_num_new_winners;
    A_size_winners_at_t(t) = new_A_w;
    B_new_winner_at_t(t) = B_num_new_winners;
    B_size_winners_at_t(t) = new_B_w;
end

       

