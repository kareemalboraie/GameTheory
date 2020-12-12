clc; close all; clear all;
n = 400; r = 0.09; new_layout = 0;
action_set_size = 2; T = 100;

if(new_layout)
    [A, W, v_Cor] = gen_graph(n, r);
    save('Layout.mat', 'A', 'W', 'v_Cor')
else
    load Layout.mat
end

% Define cost and cost vector
r = 0:n;
utility_vect = -abs(r - (n - r)).';

% Run Distributed ECFP (with fixed step-size)
Act_Mat = zeros(n, action_set_size); % Action matrix
Q = zeros(n, action_set_size);

% Initialize actions randomly 
initActions =randi(action_set_size, n);
for ii = 1:n
    Act_Mat(ii, initActions(ii)) = 1;
end
Q = Act_Mat;
Q_hat = W * Q;

avg_utility = zeros(1, T);
dist_to_CNE1 = zeros(1, T);
for t = 1:T
    fprintf('Iter = %d\n', t)
    Act_mat = zeros(n, action_set_size);
    % Do the following for every player
    % choose the next action that maximize the utility assuming all other
    % players play q_hat
    for nn = 1:n
        Pmat = compute_Pq_toy_ex(n, Q_hat(nn, :)); % P matrix has 2 rows and n/2 + 1 cols
        [~, idx] = max(Pmat * utility_vect);
        Act_mat(nn, idx) = 1;
    end
    avg_utility(t) = Q_hat(1, :) * (compute_Pq_toy_ex(n, Q_hat(1, :)) * utility_vect);
    dist_to_CNE1(t) = norm(Q_hat(1, :) - 0.5 * [1, 1]);
    % Update ech player's empirical distribution
    Qnext = Q + 1/(t + 1) * (Act_mat - Q);
    % Update the player's estimate of the 
    Q_hat = W * (Q_hat + (Qnext - Q));

    % Increment t
    Q = Qnext;
end
plot(1:T, avg_utility, 'b-')
hold on;





% Run Distributed ECFP (with discounted step-size)
Act_Mat = zeros(n, action_set_size); % Action matrix
Q = zeros(n, action_set_size);

% Initialize actions randomly 
initActions =randi(action_set_size, n);
for ii = 1:n
    Act_Mat(ii, initActions(ii)) = 1;
end
Q = Act_Mat;
Q_hat = W * Q;

avg_utility1 = zeros(1, T);
dist_to_CNE2 = zeros(1, T);
for t = 1:T
    fprintf('Iter = %d\n', t)
    Act_mat = zeros(n, action_set_size);
    % Do the following for every player
    % choose the next action that maximize the utility assuming all other
    % players play q_hat
    for nn = 1:n
        Pmat = compute_Pq_toy_ex(n, Q_hat(nn, :)); % P matrix has 2 rows and n/2 + 1 cols
        [~, idx] = max(Pmat * utility_vect);
        Act_mat(nn, idx) = 1;
    end
    avg_utility1(t) = Q_hat(1, :) * (compute_Pq_toy_ex(n, Q_hat(1, :)) * utility_vect);
    dist_to_CNE2(t) = norm(Q_hat(1, :) - 0.5 * [1, 1]);
    % Update ech player's empirical distribution
    Qnext = Q + 1/(t + 1) * (Act_mat - Q);
    % Update the player's estimate of the 
    Q_hat = W * (Q_hat + 1/(t+1) * (Qnext - Q));

    % Increment t
    Q = Qnext;
end
plot(1:T, avg_utility1, 'r-')
legend('Fixed step size', 'variable step size')
xlabel('Iteration')
ylabel('Average Utility for P1')

figure
plot(1:T, dist_to_CNE1, 'b-')
hold on
plot(1:T, dist_to_CNE2, 'r-')
legend('Fixed step size', 'variable step size')
xlabel('Iteration')
ylabel('\|\hat{q}_1 - (0.5, 0.5)\|')


function Pmat = compute_Pq_toy_ex(n, q_hat)
% First row is player always picking right
% Second row is player always picking left
% Assume n is even
Pmat = zeros(length(q_hat), n+1);
% First row if j - 1 choose right
Pmat(1, 2:end) = binopdf((0:(n-1)),n-1,q_hat(1));
Pmat(2, 1:(end-1)) = binopdf(((n-1):-1:0),n-1,q_hat(2));
end