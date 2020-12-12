clc; close all; clear all;
n = 400; r = 0.09; new_layout = 1;
channelsNum = 10; T = 100;

if(new_layout)
    [A, W, v_Cor] = gen_graph(n, r);
    save('Layout.mat', 'A', 'W', 'v_Cor')
else
    load Layout.mat
end

% Define cost and cost vector
a = [0.01, 0.1, 1, 0];
c = @(k) (a(1) * k.^3 + a(2) .* k.^2 + a(3).*k + a(4));
%c = @(k) (a(1) * k.^2 + a(2) .* k + a(3));
utility_vect = -c((0:(n-1)).');
%utility_vect2 = -c((0:(n)).');

% Run Distributed ECFP (with fixed step-size)
Act_Mat = zeros(n, channelsNum); % Action matrix
Q = zeros(n, channelsNum);

% Initialize actions randomly 
initActions =randi(channelsNum, n);
for ii = 1:n
    Act_Mat(ii, initActions(ii)) = 1;
end
Q = Act_Mat;
Q_hat = W * Q;

avg_utility = zeros(1, T);
for t = 1:T
    fprintf('Iter = %d\n', t)
    Act_mat = zeros(n, channelsNum);
    % Do the following for every player
    % choose the next action that maximize the utility assuming all other
    % players play q_hat
    for nn = 1:n
        Pmat = compute_Pq(n-1, Q_hat(nn, :)); % P matrix has m rows and n cols
        [~, idx] = max(Pmat * utility_vect);
        Act_mat(nn, idx) = 1;
    end
    avg_utility(t) = Q_hat(1, :) * (compute_Pq(n-1, Q_hat(1, :)) * utility_vect);
    % Update ech player's empirical distribution
    Qnext = Q + 1/(t + 1) * (Act_mat - Q);
    % Update the player's estimate of the 
    Q_hat = W * (Q_hat + (Qnext - Q));

    % Find average utility assuming everyone plays player 1's estimate of Q
    % bar
    %q_bar = mean(Q, 1);
    % Increment t
    Q = Qnext;
end
plot(1:T, avg_utility, 'b-')




% Run Distributed ECFP (with discounted step size)
Act_Mat = zeros(n, channelsNum); % Action matrix
Q = zeros(n, channelsNum);

% Initialize actions randomly 
initActions =randi(channelsNum, n);
for ii = 1:n
    Act_Mat(ii, initActions(ii)) = 1;
end
Q = Act_Mat;
Q_hat = W * Q;

avg_utility1 = zeros(1, T);
for t = 1:T
    fprintf('Iter = %d\n', t)
    Act_mat = zeros(n, channelsNum);
    % Do the following for every player
    % choose the next action that maximize the utility assuming all other
    % players play q_hat
    for nn = 1:n
        Pmat = compute_Pq(n-1, Q_hat(nn, :)); % P matrix has m rows and n cols
        [~, idx] = max(Pmat * utility_vect);
        Act_mat(nn, idx) = 1;
    end
    avg_utility1(t) = Q_hat(1, :) * (compute_Pq(n-1, Q_hat(1, :)) * utility_vect);
    % Update ech player's empirical distribution
    Qnext = Q + 1/(t + 1) * (Act_mat - Q);
    % Update the player's estimate of the 
    Q_hat = W * (Q_hat + 1/(t+1) * (Qnext - Q));

    % Find average utility assuming everyone plays player 1's estimate of Q
    % bar
    %q_bar = mean(Q, 1);
    % Increment t
    Q = Qnext;
end
hold on
plot(1:T, avg_utility1, 'r-')
legend('Fixed step size', 'variable step size')
xlabel('Iteration')
ylabel('Average Utility for P1')


function Pmat = compute_Pq(n, q_hat)
Pmat = zeros(length(q_hat), n+1);
for ii = 1:length(q_hat)
   Pmat(ii, :) = binopdf((0:n),n,q_hat(ii));
end
end