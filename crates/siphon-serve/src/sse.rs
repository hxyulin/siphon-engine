//! The reply loop of the routes that stream typed server-sent events, Anthropic's Messages
//! ([`crate::anthropic`]) and OpenAI's Responses ([`crate::responses`]).
//!
//! Both submit one job, feed its ids to an [`Output`] one at a time so that a reply's parts
//! keep the order the model wrote them in, and either stream each route's events or send
//! one body. A stream sends no data event before the reply's first content, so that a
//! client or proxy that times the first event measures the prefill: until then it sends
//! SSE comments after [`Reply::QUIET`] without bytes, and after it the route's
//! keep-alive.

use std::{convert::Infallible, sync::Arc, time::Duration};

use axum::{
    Json,
    response::{
        IntoResponse, Response,
        sse::{Event as SseEvent, Sse},
    },
};
use futures_util::StreamExt as _;
use serde_json::Value;
use tokio::{sync::mpsc, time::Instant};
use tokio_stream::wrappers::ReceiverStream;

use crate::{
    chat::Vocab,
    engine::{Event, Finish},
    openai::{self, ApiError, App, Delivery, Output, Prepared, RequestId, Results, submit},
    parse::Delta,
    timings::Timings,
};

/// One server-sent event: its name and its data.
pub(crate) type Frame = (&'static str, Value);

/// The name of a frame that goes out as an SSE comment.
pub(crate) const KEEP_ALIVE: &str = "";
/// The name of the one frame of a reply that is not streamed, holding its body.
const BODY: &str = "body";
/// The name of the one frame of a reply that is not streamed whose job failed, holding the
/// reason.
const FAILED: &str = "failed";

/// A reply's prompt, as its usage reports it.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Prompt {
    pub(crate) tokens: usize,
    /// Prompt tokens whose state came from the prefix cache.
    pub(crate) cached: usize,
}

/// What a route makes of one reply: its events and its body.
pub(crate) trait Reply: Send + 'static {
    /// The time without bytes after which a stream sends a keep-alive.
    const QUIET: Duration;

    /// The reply's id, for the job's log label.
    fn id(&self) -> &str;

    /// The events that open a stream, sent with its first content, or before its end or
    /// failure if it has none.
    fn start(&mut self, prompt: &Prompt) -> Vec<Frame>;

    /// The keep-alive of a stream after its start.
    fn keep_alive(&mut self) -> Frame;

    /// Adds the parts of `delta` to the reply, and to `events` the events that stream them.
    fn push(&mut self, delta: &Delta, events: &mut Vec<Frame>);

    /// The events that end a stream after the last [`Reply::push`].
    fn end(
        &mut self,
        output: &Output,
        finish: Finish,
        prompt: &Prompt,
        timings: &Timings,
    ) -> Vec<Frame>;

    /// The body of a reply that is not streamed.
    fn body(
        &mut self,
        output: &Output,
        finish: Finish,
        prompt: &Prompt,
        timings: &Timings,
    ) -> Value;

    /// The events that report a failed job, after [`Reply::start`]'s events if `started`.
    fn failure(&mut self, message: &str, prompt: &Prompt, started: bool) -> Vec<Frame>;

    /// The response to a request that is not streamed whose job failed.
    fn failed_response(message: &str) -> Response;

    /// Marks `events` just before they are sent, for a protocol whose events carry their
    /// order.
    fn number(&mut self, _events: &mut [Frame]) {}
}

/// Runs a prepared request and answers it with `reply`, streaming or not.
pub(crate) async fn respond<R: Reply>(
    app: Arc<App>,
    prepared: Prepared,
    reply: R,
    id: &RequestId,
) -> Result<Response, ApiError> {
    let Prepared {
        prompt,
        images,
        max_tokens,
        delivery,
        ignore_eos,
        samplings,
        output,
        ..
    } = prepared;
    let sampling = samplings
        .first()
        .copied()
        .ok_or_else(|| ApiError::internal("no sampling"))?;
    let label = id.label(reply.id(), 0, 1);
    let tokens = prompt.len();
    let results = submit(
        &app,
        &prompt,
        images,
        max_tokens,
        ignore_eos,
        sampling,
        &|_| Arc::clone(&label),
    )?;
    if matches!(delivery, Delivery::Whole) {
        // Runs in the handler, so that a client that leaves drops the job.
        let (out, mut rx) = mpsc::channel(1);
        let timings = run(&app.vocab, results, output, reply, tokens, out, false).await;
        let response = match rx.recv().await {
            Some((BODY, body)) => Json(body).into_response(),
            Some((_, reason)) => R::failed_response(reason.as_str().unwrap_or_default()),
            None => R::failed_response("the reply ended early"),
        };
        return Ok(match &timings {
            Some(timings) => {
                timings.log(&label);
                openai::with_server_timing(response, timings)
            }
            None => response,
        });
    }
    let (out, rx) = mpsc::channel(64);
    tokio::spawn(async move {
        if let Some(timings) = run(&app.vocab, results, output, reply, tokens, out, true).await {
            timings.log(&label);
        }
    });
    let events = ReceiverStream::new(rx).map(|frame| Ok::<_, Infallible>(sse(frame)));
    Ok(Sse::new(events).into_response())
}

/// The server-sent event of `frame`: a comment for [`KEEP_ALIVE`].
pub(crate) fn sse((name, data): Frame) -> SseEvent {
    if name == KEEP_ALIVE {
        SseEvent::default().comment("")
    } else {
        SseEvent::default().event(name).data(data.to_string())
    }
}

/// Sends one reply's frames and remembers when it last sent one, for the keep-alives.
struct Frames {
    out: mpsc::Sender<Frame>,
    last: Instant,
}

impl Frames {
    /// Sends `frames`; false once the client has left.
    async fn send(&mut self, frames: Vec<Frame>) -> bool {
        for frame in frames {
            if self.out.send(frame).await.is_err() {
                return false;
            }
            self.last = Instant::now();
        }
        true
    }
}

/// Turns the engine's events for one request into `reply`'s stream events, or, unless
/// `stream`, into one frame with the whole body, and returns the request's timings, unless
/// the job failed or the client left. Returning drops `results`, which cancels the job:
/// when the client has left or a stop ended the output.
pub(crate) async fn run<R: Reply>(
    vocab: &Vocab,
    mut results: Results,
    mut output: Output,
    mut reply: R,
    prompt_tokens: usize,
    out: mpsc::Sender<Frame>,
    stream: bool,
) -> Option<Timings> {
    let mut frames = Frames {
        out,
        last: Instant::now(),
    };
    let mut prompt = Prompt {
        tokens: prompt_tokens,
        cached: 0,
    };
    let mut started = false;
    let finish = loop {
        let mut events = Vec::new();
        let event = if stream {
            loop {
                tokio::select! {
                    event = results.next() => break event,
                    () = tokio::time::sleep_until(frames.last + R::QUIET) => {
                        let keep_alive = if started {
                            reply.keep_alive()
                        } else {
                            (KEEP_ALIVE, Value::Null)
                        };
                        if !frames.send(vec![keep_alive]).await {
                            return None;
                        }
                    }
                }
            }
        } else {
            results.next().await
        };
        match event {
            Event::Started | Event::Round | Event::Images(_) => continue,
            Event::Cached(n) => {
                prompt.cached = n;
                continue;
            }
            Event::Tokens(ids) => {
                for &id in &ids {
                    if output.done {
                        break;
                    }
                    let delta = output.push(vocab, &[id]);
                    reply.push(&delta, &mut events);
                }
            }
            Event::Done(finish) => break finish,
            Event::Failed(e) => {
                let mut frames_out = if stream {
                    reply.failure(&e, &prompt, started)
                } else {
                    vec![(FAILED, Value::String(e))]
                };
                reply.number(&mut frames_out);
                let _ = frames.send(frames_out).await;
                return None;
            }
        }
        if stream && !events.is_empty() {
            if !std::mem::replace(&mut started, true) {
                events.splice(0..0, reply.start(&prompt));
            }
            reply.number(&mut events);
            if !frames.send(events).await {
                return None;
            }
        }
        if output.done {
            break Finish::Stop;
        }
    };
    let times = results.close(finish);
    let mut events = Vec::new();
    reply.push(&output.finish(vocab, finish), &mut events);
    let timings = Timings::new(&[times], prompt.tokens, prompt.cached, output.tokens);
    if !stream {
        let body = reply.body(&output, finish, &prompt, &timings);
        return frames.send(vec![(BODY, body)]).await.then_some(timings);
    }
    if !started {
        events.splice(0..0, reply.start(&prompt));
    }
    events.extend(reply.end(&output, finish, &prompt, &timings));
    reply.number(&mut events);
    frames.send(events).await.then_some(timings)
}
