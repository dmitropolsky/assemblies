n = 1000;
k = 30;
p = 0.05;
beta = 0.05;
sigma = k*p*(1-p);
normal = makedist('Normal',k*p,sigma);
alpha = icdf(normal,(n-k)/n);
truncated = truncate(normal,alpha,inf);
stimulus = random(truncated, 1, k);
bernoulli = makedist('Binomial',1,p);
connectome = random(bernoulli,k,k);
winners = ones(1, k);
stimulus_up = stimulus*(1+beta);
normal2 = makedist('Normal', 2*k*p,2*sigma);
alpha2 = icdf(normal, (n-size(winners,2)-k)/n);
truncated2 = truncate(normal2,alpha2,inf);
newcandidates = random(truncated2,1,k);
w = size(winners,2);
supportinputs = zeros(1,w);
for i = 1:w
    supportinputs(i) = stimulus_up(i);
    for j = 1:w
        supportinputs(i) = supportinputs(i) + connectome(i,j)*winners(j);
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
    new_stimulus(i) = stimulus_up(i);
    if new_winners(i) == 1
        new_stimulus(i) = new_stimulus(i)*(1+beta);
    end
end
recurrent_inputs = zeros(1,num_new_winners);
for i = 1:num_new_winners
    b = new_winner_inputs(i);
    % Randomly generate how much came from the previous assembly
    recurrent_input = normrnd(0.5*b,sigma/sqrt(2));
    % Should we round up for better approximation?
    recurrent_input = round(recurrent_input);
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
        if new_winners(i) == 1 && winners(j) == 1
            new_connectome(i,j) = new_connectome(i,j)*(1+beta);
        end
    end
    for j = (w+1):new_w
        % Whether the new neuron j connects to neuron i
        new_connectome(i,j) = random(bernoulli);
    end
end
previous_winner_indices = find(winners == 1);
for i=1:num_new_winners
    % Working on row w+i
    recurrent_input = recurrent_inputs(i);
    inputs = randsample(previous_winner_indices, recurrent_input);
    for j=inputs
        new_connectome(w+i,j) = 1+beta;
    end
    for j = (w+1):new_w
        new_connectome(w+i,j) = random(bernoulli);
    end
end
       

