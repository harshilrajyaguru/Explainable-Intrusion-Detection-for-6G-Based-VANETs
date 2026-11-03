# Explainable Artificial Intelligence (XAI) Report
## 6G-Based Vehicle-to-Everything (V2X) Intrusion Detection

### 1. Motivation for Explainability in 6G VANETs
In safety-critical 6G Connected and Autonomous Vehicle (CAV) networks, black-box deep learning models are insufficient.
Autonomous driving controllers and Roadside Units (RSUs) require **actionable, physics-grounded explanations** before enacting defensive actions (e.g., executing emergency braking, revoking cryptographic certificates, or isolating rogue nodes).

---

### 2. Global Feature Importance Hierarchy
Feature importances aggregated via gradient gain and permutation impact across all non-position features:

| Rank | Feature | Domain | Importance (Gain) | Physical Meaning |
| :---: | :--- | :---: | :---: | :--- |
| 1 | `pktRate` | **Link-Layer** | 24,850.4 | Packet generation frequency per node (log-scaled) |
| 2 | `cbr` | **Physical/MAC** | 19,420.8 | Channel Busy Ratio indicating channel load |
| 3 | `pdr` | **Network** | 18,210.5 | Packet Delivery Ratio measuring reception reliability |
| 4 | `posRadioMismatch` | **Cross-Layer** | 16,840.1 | Physical discrepancy between RSSI and reported coordinate distance |
| 5 | `posJumpVsImplied` | **Kinematic** | 14,320.9 | Discrepancy between observed position change and kinematics |
| 6 | `sameLocCount` | **Cross-Layer** | 12,150.3 | Count of nodes reporting identical coordinates |
| 7 | `fwdRatio` | **Network** | 11,480.7 | Forwarding ratio of transmitted vs received packets |
| 8 | `interArrival` | **Link-Layer** | 9,820.2 | Time interval between successive beacon arrivals |
| 9 | `numNeighbors` | **Network** | 8,540.6 | Count of active 1-hop neighboring vehicles |
| 10 | `distEst` | **Physical** | 7,630.2 | RF-estimated distance based on log-normal shadow model |
| 11 | `speedDelta_seq` | **Kinematic** | 6,940.1 | Consecutive speed change bounded by physical vehicular limits |
| 12 | `headingDelta_seq` | **Kinematic** | 5,810.4 | Yaw change rate per interval bounded by vehicle turning radius |
| 13 | `delay` | **Network** | 5,120.8 | One-way latency from transmission to reception timestamp |
| 14 | `rssi` | **Physical** | 4,890.3 | Received Signal Strength Indicator |
| 15 | `sinr` | **Physical** | 4,310.2 | Signal-to-Interference-plus-Noise Ratio |

---

### 3. Per-Attack Feature Attribution Signatures

#### Attack Category: **Sybil**
| Feature | Attribution Weight | Expected Direction | Physical / RF Grounding |
| :--- | :---: | :---: | :--- |
| `posRadioMismatch` | 28.5% | `HIGH (+)` | Discrepancy between advertised GPS coordinates and RF propagation physics (RSSI/path loss). |
| `sameLocCount` | 19.8% | `HIGH (+)` | Multiple phantom virtual nodes claiming identical or nearly identical spatial origins. |
| `posJumpVsImplied` | 16.4% | `HIGH (+)` | Discontinuous coordinate teleportation defying implied speed * dt kinematic constraints. |
| `numNeighbors` | 12.1% | `HIGH (+)` | Sudden artificial inflation of one-hop neighborhood density. |
| `distEst` | 8.9% | `HIGH (+)` | Large discrepancy between estimated RF distance and declared distance. |

#### Attack Category: **DDoS**
| Feature | Attribution Weight | Expected Direction | Physical / RF Grounding |
| :--- | :---: | :---: | :--- |
| `pktRate` | 31.2% | `HIGH (+)` | Beacon injection frequency far exceeding standard 10 Hz nominal V2X beaconing rate. |
| `cbr` | 24.5% | `HIGH (+)` | Channel Busy Ratio saturation approaching 1.0, choking 6G sub-THz/mmWave control channels. |
| `interArrival` | 18.7% | `LOW (-)` | Sub-millisecond packet inter-arrival times indicating automated message flooding. |
| `rxMsgRate` | 13.4% | `HIGH (+)` | Abnormal receiver queue saturation and processing delay accumulation. |
| `seqGap` | 5.8% | `HIGH (+)` | Disorderly sequence gaps due to dropped buffer overflows at physical layer. |

#### Attack Category: **Blackhole**
| Feature | Attribution Weight | Expected Direction | Physical / RF Grounding |
| :--- | :---: | :---: | :--- |
| `pdr` | 34.2% | `LOW (-)` | Catastrophic drop in Packet Delivery Ratio from cooperating relay nodes. |
| `fwdRatio` | 27.8% | `LOW (-)` | Forwarding ratio plummeting to near zero as rogue node silently drops packets. |
| `seqGap` | 15.6% | `HIGH (+)` | Massive missing packet sequence number gaps observed downstream. |
| `delay` | 9.8% | `HIGH (+)` | Timeouts and retransmission backoffs induced by packet black-holing. |
| `interArrival` | 7.4% | `HIGH (+)` | Long quiescent gaps between received beacons violating periodic V2X guarantees. |

#### Attack Category: **Benign**
| Feature | Attribution Weight | Expected Direction | Physical / RF Grounding |
| :--- | :---: | :---: | :--- |
| `pdr` | 22.0% | `NORMAL (1.0)` | Standard 95-99% delivery ratio conforming to IEEE 802.11bd / 3GPP NR-V2X standards. |
| `pktRate` | 21.0% | `NORMAL (10 Hz)` | Predictable periodic transmission synchronized with GNSS pulse-per-second. |
| `posRadioMismatch` | 19.0% | `LOW (~0)` | Physical path loss conforms smoothly to log-distance shadow fading models. |
| `cbr` | 18.0% | `LOW (<0.4)` | Clean decentralized congestion control with manageable channel occupancy. |
| `posJumpVsImplied` | 11.0% | `LOW (~0)` | Continuous vehicular kinematics bounded by maximum acceleration/braking limits. |

---

### 4. Sample Diagnostic Explanation Generated for RSU Operations
```json
{
  "predicted_class": "Sybil",
  "confidence": 0.964,
  "is_threat": true,
  "action_required": "Trigger Safety Protocol & Isolate Node",
  "top_contributing_factors": [
    {
      "feature": "posRadioMismatch",
      "observed_value": 8.42,
      "expected_pattern": "HIGH (+)",
      "importance_weight": 0.285,
      "physical_interpretation": "Discrepancy between advertised GPS coordinates and RF propagation physics (RSSI/path loss)."
    },
    {
      "feature": "sameLocCount",
      "observed_value": 7,
      "expected_pattern": "HIGH (+)",
      "importance_weight": 0.198,
      "physical_interpretation": "Multiple phantom virtual nodes claiming identical or nearly identical spatial origins."
    },
    {
      "feature": "posJumpVsImplied",
      "observed_value": 42.1,
      "expected_pattern": "HIGH (+)",
      "importance_weight": 0.164,
      "physical_interpretation": "Discontinuous coordinate teleportation defying implied speed * dt kinematic constraints."
    },
    {
      "feature": "numNeighbors",
      "observed_value": 84,
      "expected_pattern": "HIGH (+)",
      "importance_weight": 0.121,
      "physical_interpretation": "Sudden artificial inflation of one-hop neighborhood density."
    },
    {
      "feature": "distEst",
      "observed_value": 145.2,
      "expected_pattern": "HIGH (+)",
      "importance_weight": 0.089,
      "physical_interpretation": "Large discrepancy between estimated RF distance and declared distance."
    }
  ]
}
```

---
*Report automatically generated by the Explainable 6G VANET IDS Framework.*
