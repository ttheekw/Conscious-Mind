// conscious routing policy, version 0 (built in).
//
// The harness interprets this script; it is not compiled in. The core model
// may replace it with a rewrite, which is adopted only if it passes the
// regression suite in src/policy.rs, and is rolled back if it misbehaves.
//
// Contract
//   estimate(c, level) -> float in 0..1
//       Chance that candidate `c` handles a task of this level well.
//   select(level, candidates) -> #{ index, reason }
//       Which candidate should run the task. `index` is a position in
//       `candidates`; `reason` is a short string.
//
//   level       1 (trivial) .. 5 (expert)
//   candidate   #{ family, tier, good, bad, calls, latency }
//       tier     1 small, 2 medium, 3 large, 0 unknown
//       good/bad arrays of 5 floats: rated outcomes per level (index level-1)
//       calls    tasks it has run; latency: average seconds, -1.0 if unknown
//
// Rhai notes: functions cannot see variables defined outside them, so shared
// numbers live inside functions. No imports, no I/O, limited steps.

// Chance of success before any evidence, by size tier and level.
fn prior(tier, level) {
    let table = [
        [0.90, 0.80, 0.50, 0.30, 0.15],
        [0.95, 0.90, 0.85, 0.65, 0.45],
        [0.97, 0.95, 0.92, 0.88, 0.85],
    ];
    let t = if tier < 1 { 2 } else if tier > 3 { 3 } else { tier };
    table[t - 1][level - 1]
}

// Beta posterior mean: the prior counts as two observations.
fn estimate(c, level) {
    let good = c.good[level - 1];
    let bad = c.bad[level - 1];
    (2.0 * prior(c.tier, level) + good) / (2.0 + good + bad)
}

// The cheapest candidate whose optimistic estimate reaches the target; if
// none does, the most promising one.
fn select(level, candidates) {
    let target = 0.75;
    let bonus = 0.15;
    let pick = -1;
    let pick_tier = 99;
    let best = 0;
    let best_mean = -1.0;
    for i in 0..candidates.len() {
        let c = candidates[i];
        let mean = estimate(c, level);
        let seen = c.good[level - 1] + c.bad[level - 1];
        // Little evidence earns a bonus, so cheaper models get tried.
        let optimistic = mean + bonus / sqrt(2.0 + seen);
        let tier = if c.tier < 1 { 2 } else { c.tier };
        if optimistic >= target && tier < pick_tier {
            pick = i;
            pick_tier = tier;
        }
        if mean > best_mean {
            best = i;
            best_mean = mean;
        }
    }
    if pick >= 0 {
        #{ index: pick, reason: "cheapest model expected to handle it" }
    } else {
        #{ index: best, reason: "no model clears the bar; using the most promising" }
    }
}
