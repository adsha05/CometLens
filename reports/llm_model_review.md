# AI Narrative Review

**Provider:** `ollama`  
**Model:** `llama3.1:8b`  
**Generated UTC:** `2026-05-27T02:18:23.724129+00:00`

> AI-generated interpretation grounded in deterministic synthetic-data monitoring artifacts.

## Executive Summary

This review covers the monitoring of a synthetic QSR purchase propensity model.

## Main Risk Drivers

- High drift in four features: merchant_novelty_rate, fuel_spend_30d, weekend_dining_frequency, and competitor_qsr_share_90d.

## Performance Interpretation

The AUC has increased by +0.0077 from the previous validation to the current period.

## Drift Interpretation

High drift is observed in four features: merchant_novelty_rate (PSI=0.5243, mean change=30.87%), fuel_spend_30d (PSI=0.1233, mean change=26.10%), weekend_dining_frequency (PSI=0.1206, mean change=-24.40%), and competitor_qsr_share_90d (PSI=0.0572, mean change=11.26%).

## Segment Interpretation

Population shifts are observed in customer segments: Loyal QSR Buyers (-2.01 percentage points), Value-Seeking Routine Shoppers (+1.21 percentage points), Unstable/New Behavior Segment (+0.59 percentage points), and Competitor Switchers (+0.21 percentage points).

## Recommended Actions

1. Consider incorporating competitor_share_rolling_60d to measure recent movement toward competing QSR merchants.
1. Measure recent movement toward competing QSR merchants with shorter and directional signals using competitor_switching_velocity.
1. Capture changing merchant behavior and distinguish genuine novelty from uncertain categorization by introducing merchant_confidence_score or merchant_descriptor_novelty_rate.
1. Track whether reduced weekend dining is recovering or persisting for purchase propensity using weekend_dining_recovery_index.

## Questions For Analyst

- Investigate the reasons behind high drift in merchant_novelty_rate and fuel_spend_30d features.
- Analyze the impact of population shifts on customer segments, particularly Loyal QSR Buyers and Value-Seeking Routine Shoppers.

## Evidence Used

- AUC increased by +0.0077 from previous validation to current period (Risk Level: High).
- High drift in four features: merchant_novelty_rate, fuel_spend_30d, weekend_dining_frequency, and competitor_qsr_share_90d.
- Population shifts observed in customer segments: Loyal QSR Buyers (-2.01 percentage points), Value-Seeking Routine Shoppers (+1.21 percentage points), Unstable/New Behavior Segment (+0.59 percentage points), and Competitor Switchers (+0.21 percentage points).

## Data Disclosure

All data used in this review is synthetic.
