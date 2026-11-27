//! Keep-alive bytes for replies that are not streamed, so that a proxy that drops a request
//! after a silence, as Cloudflare does after 100 s, waits for a long prefill or reply.
//!
//! A reply ready within [`KeepAlive::after`] goes out as it is. Past that, the response
//! commits to status 200 with a JSON content type and a chunked body: a space at once and
//! every [`KeepAlive::every`], then the reply's JSON, which JSON parsers read past the
//! leading whitespace. A reply that turns out to be an error still has status 200, since it
//! has been sent; its body is the route's error JSON, and the request's log line records
//! the real status. A client that leaves drops the body, and with it the reply's jobs.

use std::{convert::Infallible, future::Future, pin::Pin, sync::Arc, time::Duration};

use axum::{
    body::{Body, Bytes},
    http::{StatusCode, header},
    response::{IntoResponse, Response},
};
use futures_util::StreamExt as _;
use tokio::time::{Interval, MissedTickBehavior};

use crate::{log::RequestLog, openai::ErrorNote};

/// When a reply that is not streamed starts sending keep-alive bytes, and how often.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KeepAlive {
    pub after: Duration,
    pub every: Duration,
}

/// Seconds between keep-alive spaces.
pub const EVERY: Duration = Duration::from_secs(15);

/// Marks a response that the compression layer must leave alone: a compressor holds the
/// keep-alive spaces until it has enough bytes.
#[derive(Clone, Copy, Debug)]
pub struct Uncompressed;

/// A reply's future, boxed so that the body can poll it.
type Reply = Pin<Box<dyn Future<Output = Response> + Send>>;

/// The response to `reply`: as it is if it is ready within `keepalive.after`, else status 200
/// with keep-alive spaces until it is ([module docs](self)). `log` is the request's log.
pub async fn respond<E: IntoResponse>(
    keepalive: Option<KeepAlive>,
    log: &Arc<RequestLog>,
    reply: impl Future<Output = Result<Response, E>> + Send + 'static,
) -> Response {
    let reply = async move { reply.await.into_response() };
    let Some(keepalive) = keepalive else {
        return reply.await;
    };
    let mut reply: Reply = Box::pin(reply);
    tokio::select! {
        response = &mut reply => return response,
        () = tokio::time::sleep(keepalive.after) => {}
    }
    log.committed();
    let mut ticks = tokio::time::interval(keepalive.every);
    ticks.set_missed_tick_behavior(MissedTickBehavior::Delay);
    let state = Some((reply, ticks, Arc::clone(log)));
    let body = futures_util::stream::unfold(state, |state| async move {
        let (mut reply, mut ticks, log): (Reply, Interval, Arc<RequestLog>) = state?;
        tokio::select! {
            response = &mut reply => {
                let status = response.status();
                let error = response.extensions().get::<ErrorNote>().map(|e| e.0.clone());
                let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
                    .await
                    .unwrap_or_default();
                log.respond(status, error.as_deref());
                Some((Ok::<_, Infallible>(bytes), None))
            }
            _ = ticks.tick() => {
                Some((Ok(Bytes::from_static(b" ")), Some((reply, ticks, log))))
            }
        }
    });
    let mut response = (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "application/json")],
        // Fused: a layer may poll the body again after its end.
        Body::from_stream(body.fuse()),
    )
        .into_response();
    response.extensions_mut().insert(Uncompressed);
    response
}
