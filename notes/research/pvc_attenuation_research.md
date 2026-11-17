# How does PVC pipe change the sound of a leak? (literature note)

Purpose: decide how far the cast-iron leak signature (a steady 400–1,300 Hz
band about +21 dB above the logger floor) can be trusted on the 12 PVC
sites, and whether the 250–400 Hz band at U34 and U15 looks like a PVC
leak. Research only. No code, data or figures in this repo were changed.

Prepared 2026-11-17. Sources are listed in full at the end and numbered
[1]–[8] in the text.

> **Update, 2026-11-17 (later):** see `cast_iron_realism_checks.md`. The
> clips look synthesized. The U34 and U15 band has perfectly sharp edges at
> about 300 and 420 Hz, identical at both sites, and U15 also carries strong
> 120 Hz hum. A pipe-filtered PVC leak would roll off smoothly, so the
> "plausible PVC leak" reading below rests on physics that the data
> generator may not follow. The recommendation to treat PVC as low
> confidence stands, and is stronger.

---

## Bottom line

Every source we found agrees on the direction. Leak noise in plastic pipe
is lower in pitch and dies away much faster with distance than in iron
pipe. In PVC, most leak energy measured at a hydrant sat below 50 Hz [1, 2].
Hunaidi's team found that it faded at least five times faster than in metal
pipe of a similar size [2]. In iron mains measured at hydrants, leak noise
sat at about 220–1,400 Hz on average [5], which matches our three cast-iron
leaks. No published rule converts a cast-iron spectrum into a PVC one. PVC
does not slide the band down in pitch. It acts like a filter that removes
the higher pitches, and how much it removes depends mostly on how far the
leak is from the hydrant, which we do not know. Using published loss
rates, the 250–400 Hz band at U34 and U15 is consistent with a PVC leak
only if that leak is within roughly 2–13 m of the hydrant. A steady narrow
band in that range also has ordinary non-leak explanations, such as valve
or pump noise. Treat PVC predictions as low confidence, and do not apply
the 400–1,300 Hz cast-iron rule to PVC sites.

---

## 1. Key numbers by source

"Not found" means we looked for that number and did not find it in an
accessible source. We did not estimate those cells.

| # | Source | Pipe and conditions | Sensor and mounting | Leak-noise frequency | Attenuation (loss with distance) | Wave speed |
|---|---|---|---|---|---|---|
| [1] | Hunaidi & Chu 1999 (NRC Canada test site) | PVC, 150 mm, ~200 m long, soft clay, 2.4 m deep; Ottawa | Accelerometers magnetically on top of pressurized fire hydrants; hydrophones on hydrants or 19 mm service lines; hydrants ~100 m apart | Hydrophones: leak clearly above ambient at 5–50 Hz; "signal amplitudes at higher frequencies were extremely small." Accelerometers: only slightly above ambient at 50–150 Hz. Signals analysed only up to 200 Hz. | About 0.25 dB/m overall (summer). 0.33 dB/m at 15 Hz in winter (frozen topsoil). No measurable loss across pipe joints below 50 Hz. | 466–515 m/s (method- and season-dependent); 7% faster in winter |
| [2] | Hunaidi, Chu, Wang & Guan 2000 (same site, blind tests by 5 professional teams) | Same 150 mm PVC; soft silty clay | Same; sensors on hydrants worked only when hydrants were pressurized | "Mostly below 50 Hz"; best correlation band 15–100 Hz; commercial correlators failed because they defaulted to too high a band | PVC loss "higher than that for metal pipes of comparable size by a factor of at least five." Listening devices worked only within about 5 m of the leak. | As [1] |
| [3] | Brennan, Joseph, Muggleton & Gao (~2005), Southampton summary | MDPE (polyethylene), 170 mm, buried test rig, UK | Hydrophones along the pipe; loudspeaker source | Leak noise "generally detected (up to about 100 Hz)" in plastic pipe | Read off their Figure 4 (approximate): about 5 dB/m near 300 Hz, rising to about 15 dB/m near 500–600 Hz | "typically being around 400 m/s"; little effect of soil up to ~500 Hz |
| [4] | Almeida et al. 2014 | HPPE (polyethylene), 75 mm mean radius, 9.85 mm wall, ~0.8 m deep, 6 bar; Blithfield, UK | Hydrophone, geophone, accelerometer at access points 20 m and 30 m from leak | Usable band 10–28 Hz (hydrophone) to 10–104 Hz (accelerometer) for a strong leak | "about 1.6 dB/m at 150 Hz." Formula: loss (dB/m) = 8.67 × β × ω, with β = 1.99 × 10⁻⁴ s/m | 356 m/s (350–420 m/s seasonally) |
| [5] | Muntakim 2017 (thesis; field data, City of Mount Pearl, Canada; 20 sites with leaks confirmed by digging) | Ductile iron 152–305 mm, one cast iron 152 mm, copper services; 2.4–3.0 m deep; sandy crushed rock/gravel; 480–620 kPa | Sensors on fire hydrants and curb-stop valves | Leak band (high coherence) per site from 0–250 Hz up to 850–2,000 Hz; average 220–1,400 Hz. Cast-iron site: 250–1,000 Hz | Iron in the field: not found. Lab ductile iron (3 m, buried in crushed stone): 0.12 dB/m. Cites Thenikl et al. 2012: 0.6 dB/m in 100 mm steel (frequency not given) | Ductile iron 1,224–1,290 m/s; cast iron 152 mm 1,232 m/s; copper 1,164 m/s |
| [6] | Butterfield et al. 2017 (Sheffield lab) | MDPE, 63 mm (small service-size pipe), 25 m loop | Accelerometers waxed to the pipe wall, ~2.25 m from leak | Leak energy started above about 214 Hz. **With no leak**, a 318–416 Hz peak appeared at higher pressure, attributed to valve "screech" | Not given | 331 m/s (theoretical) |
| [7] | Brennan et al. 2018 (abstract only) | Plastic pipes, UK (sandy) vs Brazil (clay) sites | — | Not reported in abstract | Soil type changes the loss a lot: large in UK sandy soil, smaller in Brazilian clay | Soil shear stiffness "significantly" affects wave speed |
| [8] | Scussel et al. 2023 (abstract only) | Plastic pipes, three test sites | Hydrophones and accelerometers | Measured bandwidth is set mainly by pipe material and geometry, soil, and sensor type | Not available to us | Not available to us |

Numbers we looked for and did **not** find:

- A published ratio, transfer function or rule of thumb that maps a
  cast-iron leak spectrum onto PVC.
- Frequency-by-frequency attenuation (dB/m) for **buried cast iron** in the
  field. The only iron figures are a short lab pipe (0.12 dB/m) and a steel
  pipe cited secondhand (0.6 dB/m) [5], plus the "at least five times less
  than PVC" comparison [2].
- Any measurement of how loud a PVC leak is at its source compared with an
  iron leak. Hunaidi's team says only that plastic pipes are "quieter" [2].
- Any study using cheap clip-on loggers like ours. All the plastic-pipe
  studies used laboratory-grade hydrophones or accelerometers.

---

## 2. What the physics says, in plain terms

1. **Iron carries high pitches a long way; plastic does not.** Soft,
   flexible plastic pipe walls flex with the sound and leak energy into
   the soil and the wall itself, so the sound fades fast [1, 3]. In iron
   mains, leak noise up to 1,000–1,500 Hz still reached hydrants 100 m or
   more apart [5]. In 150 mm PVC, almost nothing above 50–150 Hz reached
   hydrants ~100 m apart [1].
2. **Loss grows with pitch.** In plastic pipe the loss in dB per metre
   rises roughly in step with frequency (the 8.67 × β × ω formula in [4]).
   Double the pitch and you lose about twice as many dB per metre. So the
   pipe acts as a **low-pass filter**, and the cut-off drops as the leak
   gets farther from the sensor [3].
3. **So PVC removes the top of the band rather than shifting it.** A PVC
   leak signal is the same broad hiss with its upper part removed. How much
   is removed depends mainly on distance, then on soil, pipe size and wall
   stiffness [3, 7, 8].
4. **Sensor type changes the apparent shape.** A pressure sensor
   (hydrophone) sees a smooth roll-off. A vibration sensor (accelerometer)
   boosts higher pitches, so the same leak shows up as a hump, a band with
   both a bottom and a top edge [3, 4]. We do not know which of these our
   loggers behave like.

---

## 3. Translating the cast-iron band to PVC

### Method

We used the loss formula from [4], loss (dB per metre) = 8.67 × β × ω, where
ω = 2π × frequency. We took a range of β (the pipe's damping factor) from
the plastic-pipe sources:

- β ≈ 2.0 × 10⁻⁴ s/m (polyethylene, measured) [4]
- β ≈ 3 × 10⁻⁴ to 5 × 10⁻⁴ s/m, from the Figure 4 readings in [3]
  (approximate)
- β ≈ 1 × 10⁻⁴ to 3 × 10⁻⁴ s/m, if Hunaidi's PVC 0.25 dB/m [1] applies
  somewhere in 15–50 Hz. That paper does not state a frequency, so this
  range is our assumption.

So we use **β ≈ 1 × 10⁻⁴ to 5 × 10⁻⁴ s/m** for buried plastic pipe. Every
number below is **our calculation** from that range, not a published
figure. PVC is stiffer than polyethylene, so it probably sits toward the
lower (less lossy) end, but we found no PVC-specific β to confirm this.

### Results

**Loss in PVC, per metre of pipe between leak and hydrant:**

| Frequency | Loss per metre (our calc.) |
|---|---|
| 50 Hz | 0.3–1.4 dB/m (fits Hunaidi's ~0.25 dB/m [1]) |
| 300 Hz | 1.6–8 dB/m |
| 400 Hz (bottom of our cast-iron band) | 2–11 dB/m |
| 1,300 Hz (top of our cast-iron band) | 7–35 dB/m |

**Highest pitch that survives with less than 20 dB of loss,** as a function
of leak-to-hydrant distance d (our calc.: f ≈ 0.37 / (β × d)):

| Leak distance d | Top of surviving band |
|---|---|
| 2 m | ~370–1,800 Hz |
| 5 m | ~150–730 Hz |
| 10 m | ~70–370 Hz |
| 25 m | ~30–150 Hz |
| 50 m | ~15–75 Hz |

### Expected PVC band

- **Pitch.** For a leak tens of metres from the hydrant, which is typical
  when hydrants are ~100 m apart [1, 5], expect energy mostly **below about
  100–150 Hz**. That matches what was measured [1, 2, 4]. Energy at
  **250–400 Hz** survives only if the leak is within roughly **2–15 m** of
  the hydrant (the closer end applies to the lossier pipe). Energy above
  ~600 Hz needs a leak within roughly 1–6 m. The cast-iron 400–1,300 Hz
  band should **not** be expected on PVC.
- **Loudness.** Over the same distance a PVC leak will be much quieter. At
  400 Hz and 10 m, the extra loss compared with iron is roughly 15–110 dB
  (our calc.). That means "gone" for most distances. There is no published
  figure for how loud the leak itself is at the pipe, so we cannot give a
  net level at the logger.
- **Shape.** If our loggers act like vibration sensors, a PVC leak would
  look like a hump peaking near f ≈ 1 / (π × β × d) (our calc. from the
  model in [4]). For example, at d = 10 m that peak falls at roughly
  60–320 Hz.

### Are U34 and U15 consistent with this?

**Partly.** The two sites are consistent with a PVC leak in two respects.
The band sits lower than the cast-iron band, and it is narrow, as
plastic-pipe leak signals tend to be [1, 3]. Under the vibration-sensor
reading above, a hump at 250–400 Hz implies a leak roughly **2–13 m** from
the hydrant (our calc.). That is possible, for example a leak on the
hydrant lateral or the nearby main.

Three things argue for caution:

- **A leak at a typical distance would sit lower.** Most of its energy
  would be below 150 Hz, and nothing at all would show at 250–400 Hz.
  U34/U15 need the leak to be unusually close.
- **Non-leak sources make steady, narrow bands in exactly this range.** In
  [6], a 318–416 Hz peak appeared with **no leak**, from a partly closed
  valve at higher pressure. In [1], spikes came from pipe and soil
  resonances, electrical mains hum, and rotating machinery such as fans
  and pumps. The hydrant itself also resonated, and its peak shifted when
  the soil froze.
- **There is no PVC leak to compare against.** We have no labelled PVC
  leak, and none of the published spectra were recorded with our logger
  model.

Our reading: U34 (+19 dB) and U15 (+13 dB) are **plausible but unconfirmed**
PVC leak candidates. The published evidence neither rules them in nor out.

---

## 4. Main uncertainties, roughly in order of impact

1. **Leak-to-hydrant distance is unknown.** It sets where the band is cut
   off, and that cut-off can move by a factor of ten between 5 m and 50 m.
   This is the biggest single unknown.
2. **Logger frequency response is unknown.** We do not know whether the
   logger hears below ~250 Hz, or whether it behaves like a pressure or a
   vibration sensor. If it is deaf below a few hundred Hz, it will miss
   most PVC leak energy (which sits below 50–150 Hz [1, 2]). Our
   cast-iron leaks start at 400 Hz, while the iron-main average starts
   lower at 220 Hz [5]. That gap hints at some low-frequency roll-off in
   the logger, or at local conditions, but this is not established.
3. **The damping factor β varies about five-fold** across plastic types,
   soils and seasons [1, 3, 4, 7]. Frozen ground increases the loss [1].
   We found no PVC-specific β.
4. **Pipe diameter and wall thickness.** These change both wave speed and
   loss [3]. Ours are unknown. Most studies used 150–170 mm pipe [1, 3].
   The one 63 mm study saw leak energy well above 200 Hz [6].
5. **Hydrant coupling.** Hydrant sensors only worked when the hydrant was
   pressurized [2]. Hydrant resonances add their own peaks [1]. Whether
   our hydrants are wet- or dry-barrel is unknown.
6. **Leak pressure and size.** Higher pressure adds high-frequency content
   [1]. Larger flow raises the level without changing the shape [1, 6].

---

## 5. How this could be used in the model

Choices for the user. None of them has been applied.

- **Do not score PVC sites with the cast-iron band rule.** A real PVC leak
  would usually fail a 400–1,300 Hz test, which would produce false "no
  leak" results. This supports reporting PVC sites separately as **low
  confidence**, consistent with assumption A8 in
  `notes/modeling/model_assumptions_checklist.md`.
- **If PVC is scored at all, use a PVC-specific search band.** Look for a
  steady, band-limited excess over the noise floor from the logger's
  lowest reliable frequency up to about **500 Hz**. Treat anything above
  ~600 Hz as unlikely to be leak noise on PVC. The band's edges are
  literature-based and approximate. They should be shown as an
  assumption, not a fitted result.
- **Before treating U34 and U15 as leak candidates, check:**
  1. Is the band smooth and a few hundred Hz wide, or a set of sharp
     lines? Lines at multiples of 60 Hz (240, 300, 360, 420 Hz) point to
     electrical hum or machinery.
  2. Do neighbouring hydrants on the same main show a weaker version of
     the same band? A real leak should fade with distance.
  3. Does the utility know of a pressure-reducing valve, pump, or
     partly closed valve near U34 or U15?
  4. What does the logger's spec sheet say about its frequency response
     below 400 Hz?
- **One informative dig.** If the budget allows one exploratory dig (the
  idea in `related_urban_labs_projects.md`), U34 is the strongest PVC
  candidate. Its result would give the first labelled PVC example.

---

## 6. Full citations

[1] Hunaidi, O., & Chu, W. T. (1999). Acoustical characteristics of leak
signals in plastic water distribution pipes. *Applied Acoustics*, 58(3),
235–254. https://doi.org/10.1016/S0003-682X(99)00013-4 — open copy (NRC
Publications Archive):
https://nrc-publications.canada.ca/eng/view/object/?id=80c56d3d-3500-455a-83a7-9eb9a94963a9

[2] Hunaidi, O., Chu, W. T., Wang, A., & Guan, W. (2000). Detecting leaks
in plastic pipes. *Journal AWWA*, 92(2), 82–94.
https://doi.org/10.1002/j.1551-8833.2000.tb08819.x — open copy (NRC
Publications Archive):
https://nrc-publications.canada.ca/eng/view/accepted/?id=1ed6ea85-fd52-4792-9f7b-425ea533b62d

[3] Brennan, M. J., Joseph, P. F., Muggleton, J. M., & Gao, Y. (c. 2005;
undated, file dated June 2005). *Some recent research results on the use of
acoustic methods to detect water leaks in buried plastic water pipes.*
Institute of Sound and Vibration Research, University of Southampton.
https://www.southampton.ac.uk/assets/imported/transforms/content-block/UsefulDownloads_Download/0994871B7147466F974C6D5A06D5E33C/Water%20Leak%20Detection%20in%20Plastic%20Pipes%5B1%5D.pdf
(summarizes Gao et al. 2004, *J. Sound Vib.* 277, 133–148; Muggleton et
al. 2004, *J. Sound Vib.* 270, 171–190; Gao et al. 2005, *J. Sound Vib.*
283, 927–941)

[4] Almeida, F., Brennan, M., Joseph, P., Whitfield, S., Dray, S., &
Paschoalini, A. (2014). On the acoustic filtering of the pipe and sensor in
a buried plastic water pipe and its effect on leak detection: An
experimental investigation. *Sensors*, 14(3), 5595–5610.
https://doi.org/10.3390/s140305595 —
https://pmc.ncbi.nlm.nih.gov/articles/PMC4004009

[5] Muntakim, A. H. (2017). *Acoustic noise characterization for leak
detection in water mains* (Master of Engineering thesis). Memorial
University of Newfoundland.
https://memorial.scholaris.ca/bitstreams/35d712a3-4d97-4f98-8f63-1c92caf77a86/download
(Chapter 3 published as Muntakim, A. H., Dhar, A. S., & Dey, R. (2017),
"Interpretation of acoustic field data for leak detection in ductile iron
and copper water distribution pipes," *Journal of Pipeline Systems
Engineering and Practice*, ASCE. We read the thesis version.)

[6] Butterfield, J. D., Collins, R. P., Krynkin, A., & Beck, S. B. M.
(2017). Experimental investigation into the influence of backfill types on
the vibro-acoustic characteristics of leaks in MDPE pipe. *Procedia
Engineering*, 186, 311–318. https://doi.org/10.1016/j.proeng.2017.03.252 —
https://eprints.whiterose.ac.uk/121851/

[7] Brennan, M. J., Karimi, M., Muggleton, J. M., Almeida, F. C. L., Kroll
de Lima, F., Ayala, P. C., Obata, D., Paschoalini, A. T., & Joseph, P. F.
(2018). On the effects of soil properties on leak noise propagation in
plastic water distribution pipes. *Journal of Sound and Vibration*, 427,
120–133. https://eprints.soton.ac.uk/419286 (abstract only; we could not
open the full text)

[8] Scussel, O., Brennan, M. J., de Almeida, F. C. L., Iwanaga, M. K.,
Muggleton, J. M., Joseph, P. F., & Gao, Y. (2023). Key factors that
influence the frequency range of measured leak noise in buried plastic
water pipes: Theory and experiment. *Acoustics*, 5(2), 490–508.
https://doi.org/10.3390/acoustics5020029 (abstract only; the publisher
site blocked automated access)

### Searched but not used

- AWWA and UK Water Industry Research (UKWIR) leakage manuals: we found no
  openly accessible copy with frequency or attenuation figures.
- Fuchs & Riehle (1991), *Applied Acoustics* 33, 1–19: we could not access
  the full text.
- Vendor frequency tables found via search (e.g., "cast iron 100–1,500 Hz,
  PVC 70–500 Hz"): left out because they are unsourced marketing figures
  and conflict with the research measurements.
