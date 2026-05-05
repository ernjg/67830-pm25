data {
    int<lower=0> K;
    int<lower=0> n_months;
    int<lower=0> N;
    vector[N] y;
    array[N] int<lower = 1, upper = n_months> month_id;
    vector[N+2] fcst;
}

parameters {
    real eta_0;
    real delta_0;
    vector[n_months] eta_month;
    vector[n_months] delta_month;
    vector[n_months] omega_month;
    vector[K] alpha;
    real phi;
    real<lower=0> sigma;
}

model {
    eta_0 ~ cauchy(0, 3);
    delta_0 ~ cauchy(0, 3);
    phi ~ cauchy(0, 1);
    eta_month ~ cauchy(0, 3);
    delta_month ~ cauchy(0, 3);
    omega_month ~ cauchy(0, .1);
    alpha ~ cauchy(0, 1);
    sigma ~ cauchy(0, 1);
    
    for (n in 1 : N - K + 1) {
        y[n] ~ cauchy(delta_0 + delta_month[month_id[n]] + ( eta_0 + eta_month[month_id[n]] ) * sin( ( (2 * pi())/(7*24) + omega_month[month_id[n]] ) * n + phi ) + dot_product(fcst[n : n + K - 1], alpha), sigma);
    }
}

generated quantities {
    vector[N] mu_pred;
    for (n in 1:K) {
        mu_pred[n] = 0;
    }
    for (n in 1 : N - K + 1) {
        {
            mu_pred[n] = delta_0 + delta_month[month_id[n]] + ( eta_0 + eta_month[month_id[n]] ) * sin( ( (2 * pi())/(7*24) + omega_month[month_id[n]] ) * n + phi ) + dot_product(fcst[n : n + K - 1], alpha);
        }
    }
}