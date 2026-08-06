# Recognition evaluation

No benchmark result is committed. The provided script turns measured pair scores and optional
latencies into true accepts, false accepts, true rejects, false rejects, precision, recall,
false acceptance rate, false rejection rate, threshold comparisons, average inference/end-to-
end latency, and approximate FPS.

## Consent-based data collection

1. Obtain written approval and voluntary consent for a small evaluation distinct from normal
   attendance. Provide a non-participation option.
2. Assign random evaluation IDs. Keep the identity key outside the repository.
3. Capture several sessions per participant across expected lighting, pose, distance, eyewear,
   and camera conditions. Do not collect more images than the approved protocol needs.
4. Create genuine pairs from different sessions of the same person and impostor pairs from
   different people. Avoid letting one prolific person dominate the pairs.
5. Record model/camera/hardware/software versions and separately measure inference and full
   event latency. Measure FPS over a defined interval after warm-up.
6. Store all real inputs below `recognition/data/evaluation/` or another ignored encrypted
   location. Never commit faces, embeddings, pair keys, or identifiable subgroup labels.
7. Review subgroup results only when consent, sample size, and privacy allow meaningful
   analysis. Small samples are uncertainty, not evidence of fairness.

## Score CSV

Copy `sample_data/evaluation_scores_template.csv` into the ignored evaluation directory:

```csv
same_person,similarity,inference_ms,end_to_end_ms
true,0.73,41.2,89.5
false,0.22,40.8,87.1
```

Rows shown above only illustrate the format; they are not project results.

## Run

```powershell
python -m recognition.scripts.evaluate recognition/data/evaluation/scores.csv `
  --threshold 0.45 --threshold 0.55 --threshold 0.65 `
  --output recognition/data/evaluation/results.json
```

The output stays ignored. Publish only reviewed aggregate results with sample counts,
confidence/uncertainty, protocol, dates, hardware, model license/version, and limitations.
Do not tune a threshold on the same pairs used for the final report.

## Interpretation

False acceptance and false rejection trade off as the threshold changes. Approximate FPS is
derived from average inference latency and does not include every pipeline cost; the separate
end-to-end measure is necessary. Attendance reliability additionally depends on detection,
tracking, temporal confirmation, cooldown, API availability, and roster correctness.

Project measurements remain unpublished until this protocol is run on documented hardware
with consented data.

`sample_data/synthetic_evaluation_fixture.csv` exists only to exercise the command and show
the input shape. Its deliberately simple numbers are not measurements and must never be
reported as project accuracy or performance. `evaluation_scores_template.csv` is the empty
header-only template for real consent-based measurements.
