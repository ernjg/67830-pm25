data {
    int<lower=0> K;
    int<lower=0> N;
    vector[N] y;
    vector[N+2] fcst;
}

parameters {
    vector[K] beta;
    vector[K] alpha;
    real mu;
    real<lower=0> sigma;
}

model {
    beta ~ cauchy(0, 1);
    alpha ~ cauchy(0, 1);
    sigma ~ cauchy(0, 1);
    mu ~ cauchy(0, 1);
    for (n in K:N) {
        y[n] ~ normal(mu + dot_product(y[n - K + 1: n], alpha) + dot_product(fcst[n - K + 3: n + 2], beta), sigma);
    }
}

generated quantities {
    vector[N] mu_pred;
    for (n in 1:(K - 1)) {
        mu_pred[n] = 0;
    }
    for (n in K:N) {
        {
            mu_pred[n] = mu + dot_product(y[n - K + 1: n], alpha) + dot_product(fcst[n - K + 3: n + 2], beta);
        }
    }
}
