"""Regenerate the measured section of README.md from outputs/summary.json.

Nothing between the MEASURED markers in README.md is typed by hand. The prose around the
numbers is written once; the numbers, and every sentence that states a direction, are
generated here from the run - with a branch wherever the run could have gone the other way.

    python tools/make_readme.py
    python tools/make_readme.py path/to/summary.json
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
README = os.path.join(HERE, "..", "README.md")
BEGIN, END = "<!-- BEGIN MEASURED -->", "<!-- END MEASURED -->"
STAGES = ("dp", "zero1", "zero2", "zero3")
LABEL = {"dp": "DP", "zero1": "ZeRO-1", "zero2": "ZeRO-2", "zero3": "ZeRO-3"}
MiB = 2**20
CAT_NAMES = {"weights": "bf16 weights", "grads": "bf16 grads", "master": "fp32 master",
             "adam_m": "Adam m", "adam_v": "Adam v", "activations": "activations",
             "gathered": "gathered weights"}


def mib(x):
    return f"{x / MiB:.3f}"


def render(R):
    L = []
    A = L.append
    sim, psi = R["config"]["sim"], R["model"]["params"]
    n, steps = sim["world"], sim["steps"]
    acc = R["acceptance"]
    gpu = R["gpu_check"]

    A(f"*Generated from `outputs/summary.json` by `tools/make_readme.py` — notebook "
      f"{R['meta']['notebook_version']}, finished {R['meta'].get('finished', '?')}, simulator on "
      f"`{R['meta']['sim_device']}`, GPU check "
      f"{'skipped (no CUDA in that run)' if gpu.get('skipped') else 'on ' + gpu['device']}.*")
    A("")
    A(f"**Acceptance: {acc['passed']}/{acc['total']} checks passed"
      + (f", {acc['skipped']} skipped" if acc["skipped"] else "") + ".**")
    A("")

    # ------------------------------------------------------------------ headlines
    A("### What the run said")
    A("")
    C = R["gates"]["C_identical"]
    all_same = all(C[s]["identical"] for s in STAGES[1:])
    m32 = R["memory_n32"]
    per0 = {s: m32[s]["persistent_per_rank"][0] for s in STAGES}
    if all_same:
        A(f"**1. Same mathematics, different memory.** On {n} ranks over {steps} steps, ZeRO-1, ZeRO-2 and "
          f"ZeRO-3 finished with fp32 master weights, bf16 working weights and per-step losses "
          f"**bit-identical** to plain data parallelism (largest difference: "
          f"{max(C[s]['max_abs_master'] for s in STAGES[1:]):.1f}), while one rank's persistent state "
          f"went from {mib(per0['dp'])} MiB to {mib(per0['zero3'])} MiB — "
          f"{per0['dp'] / per0['zero3']:.0f}× less. The loss fell {R['train']['loss_drop']:.2f} nats, so "
          f"this is a statement about a model that learned, not about untrained noise.")
    else:
        bad = [LABEL[s] for s in STAGES[1:] if not C[s]["identical"]]
        A(f"**1. The arrangements did NOT train identically:** {', '.join(bad)} differ from DP. "
          f"See the gate table below before reading anything else.")
    A("")

    B = R["gates"]["B_dp_equals_big_batch"]
    A(f"**2. Data parallelism is one big-batch GPU.** In fp64, the mean of {n} one-sequence gradients "
      f"matched the gradient of all {n * sim['micro_batch']} sequences at once to "
      f"{B['rel_diff']:.1e} relative — round-off, "
      + ("far below" if B["rel_diff"] < B["tolerance"] / 100 else "within")
      + f" the {B['tolerance']:.0e} tolerance. This is the property that lets a recipe move from one "
      f"GPU to {n} without changing what the model learns.")
    A("")

    sw = R["sweep"]
    comm = {s: sw[s][str(n)]["comm_over_P_rank0"] for s in STAGES}
    E = R["gates"]["E_comm"]
    if E["zero12_equal_dp"]:
        A(f"**3. ZeRO-1 and ZeRO-2 are free on the wire; ZeRO-3 is not.** Counted hop by hop, DP sent "
          f"{comm['dp']:.4f} P per rank per step, and ZeRO-1 and ZeRO-2 sent **exactly the same bytes, rank "
          f"for rank, at every world size**. ZeRO-3 sent {comm['zero3']:.4f} P = "
          f"{E['zero3_over_dp']:.2f}× DP. The lecture's 2P and 3P are the N→∞ limit of 2(N−1)/N and "
          f"3(N−1)/N; at N = 1 every arrangement sent 0 bytes.")
    else:
        A("**3. ZeRO-1/ZeRO-2 did not send the same bytes as DP** — see the sweep table.")
    A("")

    z3 = m32["zero3"]
    bd = z3["rank0_peak_breakdown"]
    top = max(bd, key=bd.get)
    ratio = z3["peak_per_rank"][0] / z3["persistent_per_rank"][0]
    persistent_share = sum(bd[k] for k in ("weights", "grads", "master", "adam_m", "adam_v")) / sum(bd.values())
    A(f"**4. At this scale ZeRO-3's peak is not its sharded state.** Rank 0 holds "
      f"{mib(z3['persistent_per_rank'][0])} MiB between steps but peaks at "
      f"{mib(z3['peak_per_rank'][0])} MiB ({ratio:.1f}×) during `{z3['rank0_peak_at']}`. The largest item at "
      f"that moment is **{CAT_NAMES[top]}** ({mib(bd[top])} MiB); the sharded 16 bytes are only "
      f"{persistent_share:.0%} of it. Activations are identical in all four arrangements "
      f"({mib(R['gates']['D_ledger']['activation_bytes'])} MiB), and the unit being gathered cannot be smaller "
      f"than the largest layer.")
    gaps = [sw["zero3"][str(w)]["peak_max"] - sw["zero3"][str(w)]["persistent_max"] for w in sim["sweep_worlds"]]
    if max(gaps) - min(gaps) <= 0.01 * max(gaps):
        A(f"Across the sweep that gap was **{mib(max(gaps))} MiB at every N from {sim['sweep_worlds'][0]} to "
          f"{sim['sweep_worlds'][-1]}**: sharding keeps dividing the 16 bytes, and cannot divide what sits on top of "
          f"them.")
    else:
        A(f"Across the sweep that gap ranged {mib(min(gaps))}–{mib(max(gaps))} MiB as N went from "
          f"{sim['sweep_worlds'][0]} to {sim['sweep_worlds'][-1]}.")
    A("")

    dg = R["adam_kernel_diagnostic"]
    usual = sum(dg["forms"]["usual fused form"].values())
    ours = sum(dg["forms"]["adam_ (this notebook)"].values())
    alone = sum(dg["forms"]["add_(g, alpha) alone"].values())
    if usual:
        A(f"**4b. Bit-exact needs more than elementwise.** One Adam step on {psi:,} elements, full vector "
          f"vs {n} shards, on this run's CPU (`{dg['cpu_capability']}`, {dg['threads']} threads): the usual "
          f"fused form disagreed in **{usual}** elements, `add_(g, alpha)` alone in {alone}, and this "
          f"notebook's `adam_` in **{ours}**. The first draft used the usual form and Gate C caught it as a "
          f"{dg['first_draft_gate_c_master_diff']:.1e} gap in the master weights.")
    else:
        A(f"**4b. Bit-exact needs more than elementwise.** On this run's CPU (`{dg['cpu_capability']}`) the "
          f"usual fused Adam happened to agree across shard boundaries ({usual} differing elements; "
          f"`adam_` {ours}). On the AVX-512 development machine it did not, and Gate C caught a "
          f"{dg['first_draft_gate_c_master_diff']:.1e} gap in the master weights — so `adam_` is written "
          "with separate multiply and add operations everywhere.")
    A("")

    ctl = R["gates"]["C_negative_controls"]
    small = min(ctl, key=lambda c: c["max_abs_master"])
    A(f"**5. The equality gate can fail, and catches what a loss plot hides.** Two planted bugs were both "
      f"{'caught' if all(c['caught'] for c in ctl) else 'NOT all caught'}. The subtler one, "
      f"`{small['bug']}`, changed the loss by only {small['max_abs_loss']:.1e}"
      + (" — invisible on a loss curve, because Adam divides a uniform gradient scale back out for most "
         f"weights — yet moved some individual weights by up to {small['max_abs_master']:.1e} "
         f"against a learning rate of {sim['lr']:.0e}."
         if small["max_abs_loss"] < 1e-3 else f", and moved weights by up to {small['max_abs_master']:.1e}.")
      )
    A("")

    hr = R["hot_rank"]
    wt = hr["whole-tensor"]
    units = R["model"]["units"]
    hot_desc = " and ".join(f"rank {k} ({'+'.join(v)})" for k, v in wt["hot_units"].items())
    A(f"**6. Where the cut falls decides which rank runs hot.** Cutting the flat vector at equal counts "
      f"left ZeRO-3 ranks within {hr['flat']['max_over_mean']:.3f}× of the mean. Refusing to split a tensor "
      f"put {hot_desc} at {wt['max_over_mean']:.1f}× the mean and left {wt['empty_ranks']} of {n} ranks holding "
      f"nothing. The embedding and head units are {(units[0]['end'] - units[0]['start']) / psi:.0%} and "
      f"{(units[-1]['end'] - units[-1]['start']) / psi:.0%} of the model, so no tensor boundary exists near most "
      f"of the ideal cuts — the heavy ends of the model become the heavy ranks.")
    A("")

    if gpu.get("error"):
        A(f"**7. Real GPU memory check: failed to run** on {gpu.get('device')}: `{gpu['error']}`. Every other "
          "number above comes from the CPU-side simulator and is unaffected.")
    elif gpu.get("skipped"):
        A("**7. Real GPU memory check: not run** in this pass (no CUDA). Every other number above is "
          "device-independent; run the notebook on a GPU runtime to add the allocator comparison.")
    else:
        rs = [g["persistent_ratio"] for g in gpu["stages"].values()]
        pk = {s: g["peak_ratio"] for s, g in gpu["stages"].items()}
        lo_s, hi_s = min(pk, key=pk.get), max(pk, key=pk.get)
        below = ("the simulator runs ranks one after another, so the card never holds every rank's peak at "
                 "once and the sum over ranks overstates the simultaneous peak")
        above = "kernel scratch and in-flight ring buffers, which the ledger deliberately does not count"
        if pk[lo_s] < 1 < pk[hi_s]:
            why = (f"It is reported, not gated, because two effects pull it in opposite directions: {below} "
                   f"(lowest for {LABEL[lo_s]}, {pk[lo_s]:.2f}×), while {above}, push it up (highest for "
                   f"{LABEL[hi_s]}, {pk[hi_s]:.2f}×).")
        elif pk[hi_s] <= 1:
            why = f"It sits below 1 because {below}."
        else:
            why = f"It sits above 1 because of {above}."
        A(f"**7. The counted bytes are what a real GPU allocates.** On {gpu['device']} ({gpu['work_dtype']}), "
          f"`torch.cuda.memory_allocated()` for the persistent state of all {gpu['n']} ranks came to "
          f"{min(rs):.4f}×–{max(rs):.4f}× the ledger's count across the four arrangements"
          + (" (allocator block rounding)." if max(abs(r - 1) for r in rs) <= 0.01 else " — outside 1%.")
          + f" Peak allocation ran {pk[lo_s]:.2f}×–{pk[hi_s]:.2f}× the sum of per-rank ledger peaks. " + why)
    A("")

    # ------------------------------------------------------------------ tables
    A("### Memory per rank at N = %d (counted)" % n)
    A("")
    A("| | bf16 weights | bf16 grads | fp32 master | Adam m | Adam v | **persistent** | bytes/weight counted | modelled | **peak** | peak happens at |")
    A("|---|---|---|---|---|---|---|---|---|---|---|")
    for s in STAGES:
        m = m32[s]
        h = m["rank0_steady"]
        A(f"| {LABEL[s]} | {mib(h['weights'])} | {mib(h['grads'])} | {mib(h['master'])} | {mib(h['adam_m'])} | "
          f"{mib(h['adam_v'])} | **{mib(m['persistent_per_rank'][0])}** | {m['bytes_per_weight_counted']:.4f} | "
          f"{m['bytes_per_weight_modelled']:.4f} | **{mib(m['peak_per_rank'][0])}** | `{m['rank0_peak_at']}` |")
    A("")
    A("MiB, rank 0. Counted persistent bytes equal the modelled formula *exactly* on every rank "
      f"({'yes' if R['gates']['D_ledger']['modelled_exact'] else 'NO'}).")
    A("")

    A("### The memory ladder and the wire, N = 1 … %d" % n)
    A("")
    A("| N | " + " | ".join(f"{LABEL[s]} MiB" for s in STAGES) + " | " + " | ".join(f"{LABEL[s]} ×P" for s in STAGES) + " |")
    A("|---|" + "---|" * 8)
    for w in sim["sweep_worlds"]:
        A(f"| {w} | " + " | ".join(mib(sw[s][str(w)]["persistent_max"]) for s in STAGES) + " | "
          + " | ".join(f"{sw[s][str(w)]['comm_over_P_rank0']:.4f}" for s in STAGES) + " |")
    A("")
    A("Persistent MiB on the largest rank; bytes sent per rank per step in units of P = 2Ψ.")
    A("")

    cp = R["compute_n32"]
    A("### Computation per rank per step, N = %d" % n)
    A("")
    A("| | " + " | ".join(LABEL[s] for s in STAGES) + " |")
    A("|---|---|---|---|---|")
    rows = [("unit forwards", "unit_forward", "{:.0f}"), ("unit recomputes (checkpointing)", "unit_recompute", "{:.0f}"),
            ("unit backwards", "unit_backward", "{:.0f}"), ("Adam elements updated", "adam_elements_per_rank", "{:,.0f}"),
            ("reduce-scatter calls", "reduce_scatter_calls", "{:.0f}"), ("all-gather calls", "all_gather_calls", "{:.0f}"),
            ("bytes sent (×P)", "comm_over_P", "{:.4f}")]
    for name, key, fmt in rows:
        A(f"| {name} | " + " | ".join(fmt.format(cp[s][key]) for s in STAGES) + " |")
    A("| *simulator seconds: compute / collectives / Adam* | " + " | ".join(
        "{compute:.2f} / {collectives:.2f} / {optimizer:.2f}".format(**cp[s]["sim_seconds_per_step"]) for s in STAGES) + " |")
    A("")
    A(f"Model arithmetic identical across arrangements: **{cp['model_arithmetic_identical']}**. Adam work per rank "
      f"falls {cp['adam_ratio']:.0f}× under any ZeRO stage. Simulator seconds are Python in one process — shown "
      "for completeness, not as step times.")
    A("")

    A("### Gates")
    A("")
    for c in acc["checks"]:
        mark = "✅" if c["result"] else ("⏭️ skipped" if c["result"] is None else "❌")
        A(f"- {mark} {c['name']}")
    A("")

    P = R["projection"]
    A("### Projected to V5: 30B parameters (GiB per GPU, training state only)")
    A("")
    A("| | N=8 | N=16 | N=32 | N=64 |")
    A("|---|---|---|---|---|")
    for s in STAGES:
        cells = []
        for w in (8, 16, 32, 64):
            c = P["table"][s][str(w)]
            cells.append(f"{'**' if c['fits_80gb'] else ''}{c['gib']:.1f}{'**' if c['fits_80gb'] else ''} ({c['lecture']})")
        A(f"| {LABEL[s]} | " + " | ".join(cells) + " |")
    A("")
    fit = P["smallest_fitting_world"]
    A(f"Bold = under one 80 GB card ({P['card_gib']:.1f} GiB); brackets = the lesson page. ZeRO-1's replicated "
      f"4 bytes/weight is {P['zero1_floor_gib']:.1f} GiB at 30B on every card at any N, and fills a card at "
      f"{P['zero1_boundary_params'] / 1e9:.1f}B parameters. Smallest world that fits: "
      + ", ".join(f"{LABEL[s]} {min(v) if v else 'none (at ≤64)'}" for s, v in fit.items())
      + ". Activations come on top, and finding 4 above is the reminder of how much they can be.")
    return "\n".join(L)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "outputs", "summary.json")
    with open(path, encoding="utf-8") as f:
        R = json.load(f)
    with open(README, encoding="utf-8") as f:
        text = f.read()
    head, rest = text.split(BEGIN, 1)
    _, tail = rest.split(END, 1)
    with open(README, "w", encoding="utf-8") as f:
        f.write(head + BEGIN + "\n" + render(R) + "\n" + END + tail)
    print(f"README.md measured section regenerated from {path}")


if __name__ == "__main__":
    main()
