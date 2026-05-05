data {
    int<lower=0> K;
    int<lower=0> n_months;
    int<lower=0> N;
    vector[N] y;
    array[N] int<lower = 1, upper = n_months> month_id;
    vector[N+2] fcst;
    vector[N+2] fcst2;
    vector[N+2] fcst3;
}

parameters {
    real delta_0;
    vector[n_months] delta_month;
    matrix[K, 3] alpha;
    vector[K] beta;
    real<lower=0> sigma;
}

model {
    delta_0 ~ cauchy(0, 3);
    delta_month ~ cauchy(0, 1);
    for (i in 1 : 3) {
        alpha[i] ~ cauchy(0, 1);
    }
    beta ~ cauchy(0, 1);
    sigma ~ cauchy(0, 1);
    
    for (n in 4 : N - K + 1) {
        y[n] ~ normal(delta_0 + delta_month[month_id[n]] + dot_product(fcst[n : n + K - 1], alpha[1]) + dot_product(fcst2[n : n + K - 1], alpha[2]) + dot_product(fcst3[n : n + K - 1], alpha[3]) + dot_product(y[n - K : n - 1], beta), sigma);
    }
}

generated quantities {
    vector[N] mu_pred;
    for (n in 1:K) {
        mu_pred[n] = 0;
    }
    for (n in 4 : N - K + 1) {
        {
            mu_pred[n] = delta_0 + delta_month[month_id[n]] + dot_product(fcst[n : n + K - 1], alpha[1]) + dot_product(fcst2[n : n + K - 1], alpha[2]) + dot_product(fcst3[n : n + K - 1], alpha[1]) + dot_product(y[n - K : n - 1], beta);
        }
    }
}