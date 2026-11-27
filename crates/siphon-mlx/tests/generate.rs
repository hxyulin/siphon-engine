#![cfg(all(target_os = "macos", target_arch = "aarch64"))]

use std::{env, ops::ControlFlow};

use siphon_mlx::{GenerationOptions, Model};

#[test]
#[ignore = "requires Metal, SIPHON_MODEL, and SIPHON_MLX_PYTHON"]
fn reuses_the_loaded_model_after_early_stop() {
    let model_dir = env::var_os("SIPHON_MODEL").expect("set SIPHON_MODEL to the checkpoint");
    let python =
        env::var_os("SIPHON_MLX_PYTHON").expect("set SIPHON_MLX_PYTHON to MLX-LM's Python");
    let mut model = Model::load(model_dir, python).expect("load Inkling 2 Mini on Metal");
    assert!(model.max_context() >= 32);

    let prompt = [42, 23, 5];
    let mut options = GenerationOptions::default();
    options.max_tokens = 4;
    options.temperature = 0.8;
    options.top_k = 12;
    options.top_p = 0.9;
    options.min_p = 0.05;
    options.seed = u64::MAX;

    let mut first = Vec::new();
    model
        .generate(&prompt, options, |token| {
            first.push(token);
            ControlFlow::Break(())
        })
        .expect("generate one token and release the KV cache");
    assert_eq!(first.len(), 1);

    let mut all = Vec::new();
    model
        .generate(&prompt, options, |token| {
            all.push(token);
            ControlFlow::Continue(())
        })
        .expect("generate four tokens on the same worker");
    assert_eq!(all.len(), options.max_tokens);
    assert_eq!(first, all[..1]);
}
