# Logger noise floors and sound above the floor

- Floor clips (RMS < 0.01, excluding U34): {'A': ['U05', 'U18', 'U19', 'U20', 'U21', 'U23', 'U25', 'U26', 'U30', 'U37', 'U39'], 'B': ['U01', 'U14', 'U33']}.
- Median RMS of floor clips: A 0.0045, B 0.0090 (Mann-Whitney p = 0.01). Logger B's floor spectrum sits a median +5.8 dB relative to A's.
- Floor shapes: correlation 0.99 between A and B after removing overall level; largest gap 1.8 dB at any frequency.
- Dug leaks, median dB above floor in 400-1,300 Hz: L01 +21.0, L03 +21.0, L06 +21.8.
- Clips with the 400-1,300 Hz band at least 3 dB above their floor: ['L01', 'L02', 'L03', 'L04', 'L05', 'L06', 'U02', 'U03', 'U04', 'U06', 'U07', 'U08', 'U10', 'U11', 'U12', 'U13', 'U16', 'U17', 'U22', 'U24', 'U27', 'U28', 'U29', 'U31', 'U32', 'U35', 'U36', 'U40'].
- PVC clips with any band at least 3 dB above floor: U03 (low_0_250 +22.1, edge_250_400 +19.3, leak_band_400_1300 +15.0, high_1300_2000 +16.0), U15 (edge_250_400 +12.7), U25 (low_0_250 +3.2), U28 (low_0_250 +24.5, edge_250_400 +19.5, leak_band_400_1300 +19.4, high_1300_2000 +20.0), U29 (low_0_250 +16.3, edge_250_400 +6.0, leak_band_400_1300 +9.0, high_1300_2000 +5.2), U31 (low_0_250 +18.3, edge_250_400 +8.7, leak_band_400_1300 +9.7, high_1300_2000 +8.2), U32 (low_0_250 +4.9, edge_250_400 +13.7, leak_band_400_1300 +14.9, high_1300_2000 +15.0), U34 (low_0_250 +5.5, edge_250_400 +19.0).

Band-level table: `excess_over_noise_floor_by_band.csv`.
