n = 10000;
k = 100;
p = 0.01;
% will use betas: 0.001, 0.003, 0.007, 0.01, 0.05, 0.1, 0.5
beta = 0.05;
normal = makedist('Binomial',k,p);
alpha = icdf(normal,(n-k)/n);
truncated = truncate(normal,alpha,inf);
stimulus = random(truncated, 1, k);
stimulus = stimulus*(1+beta);
bernoulli = makedist('Binomial',1,p);
connectome = random(bernoulli,k,k);
% set of indices of previous round winners
winners = zeros(0,0);
winners = [winners [1:k]];

normal2 = makedist('Binomial', 2*k,p);
alpha2 = icdf(normal2, (n-k)/n);  % should it be (n-k-w)/n
truncated2 = truncate(normal2,alpha2,inf);

T=30;
new_winner_at_t = zeros(1,T);
support_set = winners;
size_winners_at_t = zeros(1,T);
% Going in at each stage we have:
% winners: set of indices of previous round winners
% connectome: w x w matrix of synapses between neurons in winners.
% stimulus: array of total input weight from stimulus to each winner.
for t=1:T
    w = size(connectome,1);
    newcandidates = random(truncated2,1,k);
    supportinputs = zeros(1,w);
    for i = 1:w
        supportinputs(i) = stimulus(i);
    end
    for i = winners
        for j = 1:w
            supportinputs(j) = supportinputs(j) + connectome(i,j);
        end
    end
    both = [supportinputs newcandidates];
    [~,I] = maxk(both,k);
    num_new_winners = nnz(I > w);
    num_support_winners = k - num_new_winners;
    new_w = w+num_new_winners;
    new_winner_inputs = zeros(1,num_new_winners);
    new_winners = zeros(0,0);
    j=1;
    for i = I
        if i > w
            new_winner_inputs(j) = both(i);
            new_winners = [new_winners (w+j)];
            j = j+1;
        else
            new_winners = [new_winners i];
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
    end
    for i = new_winners
        new_stimulus(i) = new_stimulus(i)*(1+beta);
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
            new_connectome(i,j) = connectome(i,j);
        end
        if ~ismember(i,winners)  %not a winner, input to new winner random
            for j=(w+1):new_w
                new_connectome(i,j) = random(bernoulli);
            end
        end
    end
    for i=(w+1):new_w
        for j=1:new_w
            % Whether the new neuron i connects into neuron j
            new_connectome(i,j) = random(bernoulli);
        end
    end
    for i=winners
        for j=new_winners
            new_connectome(i,j) = new_connectome(i,j)*(1+beta);
        end
    end
    for i=1:num_new_winners
        % Working on row w+i
        recurrent_input = recurrent_inputs(i);
        inputs = randsample(winners, recurrent_input);
        for j=inputs
            new_connectome(j,w+i) = 1+beta;
        end
    end
    new_winner_at_t(t) = num_new_winners;
    support_set = union(support_set, new_winners);
    size_winners_at_t(t) = size(support_set,2);
    winners = new_winners;
    connectome = new_connectome;
    stimulus = new_stimulus;
end
densities = zeros(1,new_w);
for i=1:new_w
    num_edges = nnz(connectome(:,i) > 0);
    densities(i) = num_edges/new_w;
end
       

