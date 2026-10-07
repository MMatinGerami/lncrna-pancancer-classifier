// Processes for main.nf. Python runs through params.python (uv by default), so every task
// uses the locked project environment. Each script is a process input, so editing it
// invalidates the cached tasks that ran it (-resume).

process TOY_DATA {
    input:
    path script

    output:
    path 'processed'

    script:
    """
    ${params.python} ${script} processed
    """
}

process BENCHMARK {
    tag "${universe} ${model} seed ${seed}"
    cpus { model == 'xgboost' ? 6 : model == 'mlp' ? 2 : 4 }
    memory { 6.GB * task.attempt }

    input:
    tuple val(universe), val(model), val(seed), path(processed)
    path config
    path script

    output:
    path "${universe}__${model}__seed${seed}"

    script:
    """
    # keep XGBoost (OpenMP) and GridSearchCV (joblib) inside the cores given to this task
    export OMP_NUM_THREADS=${task.cpus} LOKY_MAX_CPU_COUNT=${task.cpus}
    ${params.python} ${script} \\
        --universes ${universe} --models ${model} --seed ${seed} \\
        --config ${config} --processed ${processed} \\
        --results ${universe}__${model}__seed${seed}
    """
}

process SEED_STABILITY {
    input:
    path runs
    path script

    output:
    path 'seed_stability'

    script:
    """
    ${params.python} ${script} ${runs} --out seed_stability
    """
}
