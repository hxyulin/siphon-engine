//! One frame: the plan as a bar over the GPU's memory, a table of its parts, the
//! configuration and the longest context of every configuration.

use ratatui::{
    buffer::Buffer,
    layout::{Constraint, Layout, Rect},
    style::{Color, Modifier, Style, Stylize as _},
    text::{Line, Span},
    widgets::{Block, BorderType, Cell, Paragraph, Row, Table, Widget as _},
};
use siphon::{KvFormat, WeightFormat, memory::gib};

use crate::{
    app::{App, FORMATS},
    device::Source,
};

/// Rows a printed frame takes.
const HEIGHT: u16 = 31;

/// Rows of a printed frame: one more for the vision tower's part.
pub fn height(app: &App) -> u16 {
    HEIGHT + u16::from(app.vision)
}

const DIM: Color = Color::Rgb(110, 116, 126);
const ACCENT: Color = Color::Rgb(122, 162, 247);
const GOOD: Color = Color::Rgb(115, 201, 145);
const BAD: Color = Color::Rgb(240, 113, 120);

/// One part of the GPU's memory.
struct Part {
    name: &'static str,
    bytes: usize,
    color: Color,
    symbol: &'static str,
    grows: String,
}

impl Part {
    fn new(name: &'static str, bytes: usize, color: Color, grows: impl Into<String>) -> Self {
        Self {
            name,
            bytes,
            color,
            symbol: "█",
            grows: grows.into(),
        }
    }

    fn symbol(self, symbol: &'static str) -> Self {
        Self { symbol, ..self }
    }
}

pub fn draw(app: &App, area: Rect, buf: &mut Buffer) {
    let [header, bar, middle, footer] = Layout::vertical([
        Constraint::Length(4),
        Constraint::Length(7),
        Constraint::Min(16),
        Constraint::Length(1),
    ])
    .areas(area);
    let [table, side] =
        Layout::horizontal([Constraint::Percentage(50), Constraint::Percentage(50)]).areas(middle);
    let [config, contexts] =
        Layout::vertical([Constraint::Length(8), Constraint::Min(8)]).areas(side);

    let parts = parts(app);
    draw_header(app, header, buf);
    draw_bar(app, &parts, bar, buf);
    draw_table(app, &parts, table, buf);
    draw_config(app, config, buf);
    draw_contexts(app, contexts, buf);
    draw_footer(footer, buf);
}

fn block(title: &str) -> Block<'_> {
    Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(Style::new().fg(DIM))
        .title(Line::from(format!(" {title} ")).bold().fg(ACCENT))
}

/// 1234567 as "1,234,567".
fn thousands(n: usize) -> String {
    let digits = n.to_string();
    let mut out = String::new();
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    out
}

fn format_name(weights: WeightFormat, kv: KvFormat) -> String {
    let weights = if weights == WeightFormat::Fp8 {
        "FP8"
    } else {
        "BF16"
    };
    let kv = match kv {
        KvFormat::Fp8 => "FP8",
        KvFormat::Bf16 => "BF16",
        KvFormat::Nvfp4 => "NVFP4",
        KvFormat::Nvfp4All => "NVFP4-all",
        KvFormat::Mixed => "mixed",
    };
    format!("{weights} / {kv}")
}

/// The GPU's memory by holder, in the order the bar draws it; the vision tower only with
/// `--vision`.
fn parts(app: &App) -> Vec<Part> {
    let mut parts = all_parts(app);
    if !app.vision {
        parts.retain(|p| p.name != "Vision");
    }
    parts
}

#[expect(clippy::too_many_lines, reason = "one entry per holder")]
fn all_parts(app: &App) -> Vec<Part> {
    let config = app.config();
    let (context, _) = app.context();
    let plan = app.plan();
    let one = config.planner.plan(1);
    let zero = config.planner.plan(0);
    let per_position =
        |bytes: usize| format!("{} KiB × {}", bytes.div_ceil(1024), thousands(context));
    let left = app.device.free.saturating_sub(plan.total());
    let margin = left.min(app.device.free - app.budget.min(app.device.free));
    let margin_name = if app.limit.is_some() && app.budget + app.margin < app.device.free {
        "Past --memory"
    } else {
        "Safety margin"
    };
    vec![
        Part::new(
            "Driver, others",
            app.device.others,
            Color::Rgb(74, 78, 86),
            "reserve, programs",
        ),
        Part::new(
            "CUDA context",
            app.device.context,
            Color::Rgb(120, 126, 138),
            "measured",
        ),
        Part::new(
            "Weights",
            plan.weights,
            Color::Rgb(86, 136, 245),
            "checkpoint",
        ),
        Part::new(
            "FP8 copies",
            plan.fp8_copies,
            Color::Rgb(137, 180, 250),
            "decode matrices",
        ),
        Part::new(
            if plan.pooled { "KV pool" } else { "KV cache" },
            plan.kv,
            Color::Rgb(80, 200, 140),
            if plan.pooled {
                format!("{} + rings", thousands(context))
            } else {
                per_position(one.kv)
            },
        ),
        Part::new(
            "Draft KV cache",
            plan.draft_kv,
            Color::Rgb(166, 227, 161),
            per_position(one.draft_kv - zero.draft_kv),
        ),
        Part::new(
            "Draft",
            plan.draft,
            Color::Rgb(198, 140, 250),
            "weights, buffers",
        ),
        Part::new(
            "Prefill",
            plan.prefill,
            Color::Rgb(245, 194, 90),
            format!("{}-row chunk", thousands(app.prefill_chunk)),
        ),
        Part::new(
            "Verify",
            plan.verify,
            Color::Rgb(250, 150, 90),
            "8 rows, speculator",
        ),
        Part::new(
            "Batched step",
            plan.batch,
            Color::Rgb(235, 160, 172),
            format!("{} rows", app.streams),
        ),
        Part::new(
            if plan.pooled {
                "Conv states"
            } else {
                "Rings, conv"
            },
            plan.rings,
            Color::Rgb(80, 196, 205),
            if plan.pooled {
                "per stream"
            } else {
                "windows, conv states"
            },
        ),
        Part::new(
            "Prefix cache",
            plan.prefix,
            Color::Rgb(148, 226, 213),
            if plan.pooled {
                "copies of conv states"
            } else {
                "copies of rings, conv"
            },
        ),
        Part::new(
            "Vision",
            plan.vision,
            Color::Rgb(242, 205, 205),
            "tower, largest image",
        ),
        Part::new(
            "Runtime",
            plan.runtime,
            Color::Rgb(170, 174, 182),
            "modules, graphs",
        ),
        Part::new(margin_name, margin, Color::Rgb(150, 120, 80), "left free").symbol("░"),
        Part::new(
            "Unused",
            left - margin,
            DIM,
            if left - margin >= app.device.context / 2 {
                "room for more".to_owned()
            } else {
                "rounding".to_owned()
            },
        )
        .symbol("·"),
    ]
}

fn draw_header(app: &App, area: Rect, buf: &mut Buffer) {
    let device = &app.device;
    let source = match device.source {
        Source::NvidiaSmi => "nvidia-smi",
        Source::Given => "--free",
        Source::Assumed => "assumed idle, no nvidia-smi",
    };
    let mut budget = vec![
        Span::raw("budget "),
        Span::raw(format!("{} GiB", gib(app.budget))).bold(),
        Span::raw(format!(
            ": {} GiB free less a {} GiB margin",
            gib(device.free),
            gib(app.margin)
        ))
        .fg(DIM),
    ];
    if let Some(limit) = app.limit {
        budget.push(
            Span::raw(format!(
                ", within --memory {} GiB less the context",
                gib(limit)
            ))
            .fg(DIM),
        );
    }
    let lines = vec![
        Line::from(vec![
            Span::raw(device.name.clone()).bold(),
            Span::raw(format!(
                "  {} GiB, {} GiB free after the CUDA context ",
                gib(device.total),
                gib(device.free)
            )),
            Span::raw(format!("({source})")).fg(DIM),
        ]),
        Line::from(budget),
    ];
    Paragraph::new(lines)
        .block(
            block(&format!("Siphon memory plan · {}", app.model))
                .padding(ratatui::widgets::Padding::horizontal(1)),
        )
        .render(area, buf);
}

/// Splits `width` cells among `parts` by bytes, at least one cell for each part that has
/// any, largest remainders first.
fn cells(parts: &[Part], width: usize) -> Vec<usize> {
    let total: usize = parts.iter().map(|p| p.bytes).sum();
    if total == 0 || width == 0 {
        return vec![0; parts.len()];
    }
    let nonzero = parts.iter().filter(|p| p.bytes > 0).count();
    let spare = width.saturating_sub(nonzero);
    let exact: Vec<f64> = parts
        .iter()
        .map(|p| p.bytes as f64 / total as f64 * spare as f64)
        .collect();
    #[expect(clippy::cast_sign_loss, reason = "shares are positive")]
    let mut out: Vec<usize> = parts
        .iter()
        .zip(&exact)
        .map(|(p, &e)| usize::from(p.bytes > 0) + e.floor() as usize)
        .collect();
    let mut order: Vec<usize> = (0..parts.len()).filter(|&i| parts[i].bytes > 0).collect();
    order.sort_by(|&a, &b| (exact[b] - exact[b].floor()).total_cmp(&(exact[a] - exact[a].floor())));
    let used: usize = out.iter().sum();
    for &i in order.iter().take(width.saturating_sub(used)) {
        out[i] += 1;
    }
    out
}

fn draw_bar(app: &App, parts: &[Part], area: Rect, buf: &mut Buffer) {
    let plan = app.plan();
    let (context, most) = app.context();
    let fits = plan.total() <= app.budget;
    let status = if fits {
        Span::raw("fits").fg(GOOD).bold()
    } else {
        Span::raw(format!(
            "{} GiB over the budget",
            gib(plan.total() - app.budget)
        ))
        .fg(BAD)
        .bold()
    };
    let summary = Line::from(vec![
        Span::raw(format!("{} GiB", gib(plan.total()))).bold(),
        Span::raw(" for "),
        Span::raw(thousands(context)).bold(),
        Span::raw(if !most {
            " positions · "
        } else if context == 0 {
            " positions: no context fits · "
        } else {
            " positions, the most that fit · "
        }),
        status,
    ]);
    let inner_width = usize::from(area.width.saturating_sub(4));
    let widths = cells(parts, inner_width);
    let bar: Line = parts
        .iter()
        .zip(&widths)
        .map(|(p, &w)| Span::styled(p.symbol.repeat(w), Style::new().fg(p.color)))
        .collect();
    let total: usize = parts.iter().map(|p| p.bytes).sum();
    let mut ruler = vec![' '; inner_width];
    let mut gib_mark = 0;
    while gib_mark << 30 <= total {
        let x = (gib_mark << 30) * inner_width / total;
        let label = if gib_mark == 0 {
            "0".to_owned()
        } else {
            gib_mark.to_string()
        };
        if x + label.len() <= inner_width {
            for (i, c) in label.chars().enumerate() {
                ruler[x + i] = c;
            }
        }
        gib_mark += 8;
    }
    let unit = " GiB";
    if let Some(end) = ruler.iter().rposition(|c| *c != ' ')
        && end + 1 + unit.len() <= inner_width
    {
        for (i, c) in unit.chars().enumerate() {
            ruler[end + 1 + i] = c;
        }
    }
    let ruler: String = ruler.into_iter().collect();
    let lines = vec![
        summary,
        Line::default(),
        bar.clone(),
        bar,
        Line::from(Span::raw(ruler).fg(DIM)),
    ];
    let device = app.device.others + app.device.context + app.device.free;
    let title = format!("GPU memory, {} GiB", gib(device));
    Paragraph::new(lines)
        .block(block(&title).padding(ratatui::widgets::Padding::horizontal(1)))
        .render(area, buf);
}

fn draw_table(app: &App, parts: &[Part], area: Rect, buf: &mut Buffer) {
    let total: usize = parts.iter().map(|p| p.bytes).sum::<usize>().max(1);
    let rows: Vec<Row> = parts
        .iter()
        .map(|p| {
            let style = if p.bytes == 0 {
                Style::new().fg(DIM)
            } else {
                Style::new()
            };
            let share = p.bytes as f64 / total as f64 * 100.0;
            Row::new(vec![
                Cell::from(Span::styled(
                    if p.symbol == "█" { "■" } else { p.symbol },
                    Style::new().fg(p.color),
                )),
                Cell::from(p.name),
                Cell::from(Line::from(gib(p.bytes)).right_aligned()),
                Cell::from(Line::from(format!("{share:.1}%")).right_aligned()),
                Cell::from(Span::raw(p.grows.clone()).fg(DIM)),
            ])
            .style(style)
        })
        .collect();
    let header = Row::new(vec![
        Cell::from(""),
        Cell::from("Part"),
        Cell::from(Line::from("GiB").right_aligned()),
        Cell::from(Line::from("Share").right_aligned()),
        Cell::from("Sized by"),
    ])
    .style(Style::new().fg(DIM).add_modifier(Modifier::BOLD));
    let plan = app.plan();
    let title = format!("Breakdown, {} GiB planned", gib(plan.total()));
    Table::new(
        rows,
        [
            Constraint::Length(1),
            Constraint::Length(15),
            Constraint::Length(6),
            Constraint::Length(6),
            Constraint::Min(10),
        ],
    )
    .header(header)
    .column_spacing(1)
    .block(block(&title))
    .render(area, buf);
}

/// A row of choices with the chosen one highlighted.
fn choices<'a>(label: &'a str, options: &[String], chosen: usize, key: &'a str) -> Line<'a> {
    let mut spans = vec![Span::raw(format!("{label:<10}")).fg(DIM)];
    for (i, option) in options.iter().enumerate() {
        if i > 0 {
            spans.push(Span::raw(" "));
        }
        let span = Span::raw(format!(" {option} "));
        spans.push(if i == chosen {
            span.bold().fg(Color::Black).bg(ACCENT)
        } else {
            span.fg(DIM)
        });
    }
    spans.push(Span::raw(format!("  {key}")).fg(DIM));
    Line::from(spans)
}

fn draw_config(app: &App, area: Rect, buf: &mut Buffer) {
    let config = app.config();
    let fp8 = |b: bool| usize::from(!b);
    let pair = |a: &str, b: &str| vec![a.to_owned(), b.to_owned()];
    let drafts: Vec<String> = app.drafts.iter().map(|d| d.name().to_owned()).collect();
    let (context, most) = app.context();
    let per_position = config.planner.bytes_per_position();
    let lines = vec![
        choices(
            "Weights",
            &pair("FP8", "BF16"),
            fp8(config.weights == WeightFormat::Fp8),
            "w",
        ),
        choices(
            "KV cache",
            &pair("FP8", "BF16"),
            fp8(config.kv == KvFormat::Fp8),
            "k",
        ),
        choices("Draft", &drafts, app.draft, "d"),
        Line::from(vec![
            Span::raw(format!("{:<10}", "Context")).fg(DIM),
            Span::raw(thousands(context)).bold(),
            Span::raw(if most { "  most that fit" } else { "" }).fg(DIM),
            Span::raw("  ←→ ↑↓ m").fg(DIM),
        ]),
        Line::from(vec![
            Span::raw(format!("{:<10}", "Position")).fg(DIM),
            Span::raw(format!("{} KiB", per_position.div_ceil(1024))).bold(),
            Span::raw(format!(
                "  {} {}",
                if app.streams > 1 {
                    "per-seq limit"
                } else {
                    "model limit"
                },
                thousands(config.planner.max_positions())
            ))
            .fg(DIM),
        ]),
    ];
    Paragraph::new(lines)
        .block(block("Configuration").padding(ratatui::widgets::Padding::horizontal(1)))
        .render(area, buf);
}

fn draw_contexts(app: &App, area: Rect, buf: &mut Buffer) {
    let header = std::iter::once(Cell::from("Formats"))
        .chain(
            app.drafts
                .iter()
                .map(|d| Cell::from(Line::from(d.name()).right_aligned())),
        )
        .collect::<Row>()
        .style(Style::new().fg(DIM).add_modifier(Modifier::BOLD));
    let rows: Vec<Row> = FORMATS
        .iter()
        .enumerate()
        .map(|(f, &(weights, kv))| {
            std::iter::once(Cell::from(format_name(weights, kv)))
                .chain(app.configs.iter().enumerate().map(|(d, configs)| {
                    let text = app
                        .max_context(&configs[f])
                        .map_or_else(|| "none".to_owned(), thousands);
                    let line = Line::from(text).right_aligned();
                    if d == app.draft && f == app.format {
                        Cell::from(line.bold().fg(Color::Black).bg(ACCENT))
                    } else {
                        Cell::from(line)
                    }
                }))
                .collect::<Row>()
        })
        .collect();
    let widths = std::iter::once(Constraint::Length(15))
        .chain(app.drafts.iter().map(|_| Constraint::Length(10)))
        .collect::<Vec<_>>();
    Table::new(rows, widths)
        .header(header)
        .column_spacing(1)
        .block(block("Longest context").padding(ratatui::widgets::Padding::horizontal(1)))
        .render(area, buf);
}

fn draw_footer(area: Rect, buf: &mut Buffer) {
    let keys = [
        ("w", "weights"),
        ("k", "KV cache"),
        ("d", "draft"),
        ("←→", "±8K"),
        ("↑↓", "±64K"),
        ("m", "most that fit"),
        ("q", "quit"),
    ];
    let spans: Vec<Span> = keys
        .iter()
        .flat_map(|(key, what)| {
            [
                Span::raw(format!(" {key} ")).bold().fg(ACCENT),
                Span::raw(format!("{what}  ")).fg(DIM),
            ]
        })
        .collect();
    Paragraph::new(Line::from(spans)).render(area, buf);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn part(bytes: usize) -> Part {
        Part {
            name: "",
            bytes,
            color: DIM,
            symbol: "█",
            grows: String::new(),
        }
    }

    #[test]
    fn cells_fill_the_width_and_show_small_parts() {
        let parts = [part(1000), part(1), part(0), part(500)];
        let widths = cells(&parts, 40);
        assert_eq!(widths.iter().sum::<usize>(), 40);
        assert_eq!(widths[1], 1, "a tiny part still gets a cell");
        assert_eq!(widths[2], 0, "an empty part gets none");
        assert!(widths[0] > widths[3]);
    }

    #[test]
    fn thousands_groups_digits() {
        assert_eq!(thousands(0), "0");
        assert_eq!(thousands(999), "999");
        assert_eq!(thousands(1000), "1,000");
        assert_eq!(thousands(1_048_576), "1,048,576");
    }
}
