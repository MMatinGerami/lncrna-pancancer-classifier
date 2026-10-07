#!/usr/bin/env nextflow
/*
 * Seed-stability benchmark.
 *
 * 02_benchmark.py reports one training seed. This pipeline runs every
 * gene universe x model x seed combination as its own task, then pools them
 * (18_seed_stability.py) to ask whether seed-to-seed spread is small next to the
 * test-set bootstrap interval, and whether the lncRNA vs protein-coding gap keeps
 * its sign. The test split itself is fixed by 01_prepare_data.py.
 *
 *   nextflow run . -profile test                      # synthetic data, a few minutes
 *   nextflow run . --seeds 1,2,3,4,5 --models logreg  # real data in data/processed
 *   nextflow run . -resume                            # rerun only what changed
 */

include { TOY_DATA; BENCHMARK as BENCHMARK_CPU; BENCHMARK as BENCHMARK_GPU; SEED_STABILITY } from './modules/benchmark'

workflow {
    main:
    def scripts = "${projectDir}/scripts"
    def processed = params.toy
        ? TOY_DATA(file("${scripts}/make_toy_data.py"))
        : channel.fromPath(params.processed, type: 'dir', checkIfExists: true)

    def combos = channel.fromList(params.universes.tokenize(','))
        .combine(channel.fromList(params.models.tokenize(',')))
        .combine(channel.fromList(params.seeds.toString().tokenize(',')).map { s -> s as Integer })
        .combine(processed)

    // the MLP trains on the Apple GPU, which one task at a time can use (maxForks in
    // nextflow.config); logreg and XGBoost share the CPU cores
    def by_device = combos.branch { _universe, model, _seed, _dir ->
        gpu: model == 'mlp'
        cpu: true
    }
    def config = file(params.config, checkIfExists: true)
    def benchmark = file("${scripts}/02_benchmark.py")
    def runs = BENCHMARK_CPU(by_device.cpu, config, benchmark)
        .mix(BENCHMARK_GPU(by_device.gpu, config, benchmark))

    SEED_STABILITY(runs.collect(), file("${scripts}/18_seed_stability.py"))
}
