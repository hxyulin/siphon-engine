use std::{
    env,
    ops::ControlFlow,
    sync::{Arc, PoisonError, mpsc::sync_channel},
    thread,
};

use anyhow::{Context as _, ensure};
use siphon_mlx::{GenerationOptions, Model};

use super::{CANCELLED, Control, Engine, Event, Finish, Job, Options};
use crate::serve::{
    log::log,
    metrics::{SchedulerMetrics, StreamPhase},
};

const DEFAULT_CONTEXT: usize = 8192;

pub(in crate::serve) fn start(
    options: &Options,
    stop: Vec<u32>,
    _warmup: Vec<u32>,
) -> anyhow::Result<Engine> {
    let queue_len = options.queue;
    let (jobs, queue) = sync_channel(queue_len.checked_add(1).context("queue is too large")?);
    let python = env::var_os("SIPHON_MLX_PYTHON").unwrap_or_else(|| "python3".into());
    let mut model = Model::load(&options.model_dir, python).context(
        "loading MLX model (set SIPHON_MLX_PYTHON to a Python executable with Inkling 2 MLX-LM support)",
    )?;
    let max_context = options
        .capacity
        .unwrap_or_else(|| DEFAULT_CONTEXT.min(model.max_context()));
    ensure!(
        max_context <= model.max_context(),
        "--context {max_context} exceeds the checkpoint limit of {}",
        model.max_context()
    );

    let control = Arc::new(Control::default());
    let engine_control = Arc::clone(&control);
    let greedy = options.greedy;
    let mut generation = GenerationOptions::default();
    if let Some(chunk) = options.prefill_chunk {
        generation.prefill_chunk = chunk;
    }
    thread::Builder::new()
        .name("mlx-engine".to_owned())
        .spawn(move || {
            loop {
                let stopping = engine_control
                    .stopped
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .take();
                if let Some(report) = stopping {
                    drop(model);
                    let _ = report.send("MLX worker stopped".to_owned());
                    break;
                }
                let Ok(submission) = queue.recv() else {
                    break;
                };
                for job in submission {
                    run_job(
                        &mut model,
                        job,
                        &stop,
                        generation,
                        greedy,
                        &engine_control.metrics,
                    );
                }
            }
        })?;
    log!("MLX ready with one sequence and {max_context} context positions");
    Ok(Engine {
        jobs,
        max_context,
        load: Arc::default(),
        queue: queue_len,
        streams: 1,
        control,
    })
}

fn run_job(
    model: &mut Model,
    mut job: Job,
    stop: &[u32],
    mut options: GenerationOptions,
    greedy: bool,
    metrics: &Arc<SchedulerMetrics>,
) {
    if job.events.is_closed() {
        return;
    }
    if job.cancelled() {
        let _ = job.events.send(Event::Failed(CANCELLED.to_owned()));
        return;
    }
    job.start();
    if job.events.send(Event::Started).is_err() {
        return;
    }
    options.max_tokens = job.max_tokens;
    options.temperature = if greedy {
        0.0
    } else {
        job.sampling.temperature
    };
    options.top_k = job.sampling.top_k;
    options.top_p = job.sampling.top_p;
    options.min_p = job.sampling.min_p;
    options.seed = job.sampling.seed;

    let mut finish = Finish::Length;
    let mut stream = metrics.start();
    let mut remaining = job.max_tokens;
    let result = model.generate_with_progress(
        &job.prompt,
        options,
        |tokens| metrics.prefill(tokens),
        |id, elapsed| {
            let connected = !job.cancelled() && !job.events.is_closed();
            let eos = !job.ignore_eos && stop.contains(&id);
            let emit = connected && !eos;
            if remaining < job.max_tokens {
                metrics.decode(1, usize::from(emit), elapsed);
            }
            remaining -= 1;
            stream.set((emit && remaining > 0).then_some(StreamPhase::Decode));
            if connected && eos {
                finish = Finish::Stop;
            }
            if emit && job.events.send(Event::from_tokens(&[id])).is_ok() {
                ControlFlow::Continue(())
            } else {
                ControlFlow::Break(())
            }
        },
    );
    drop(stream);
    let end = if job.cancelled() {
        Event::Failed(CANCELLED.to_owned())
    } else {
        match result {
            Ok(()) => Event::Done(finish),
            Err(error) => Event::Failed(error.to_string()),
        }
    };
    let _ = job.events.send(end);
}
