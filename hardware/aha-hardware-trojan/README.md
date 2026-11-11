# AHA! — LLM-Assisted Hardware Trojan Design on an AES Core

Hardware Trojans designed with an LLM (Gemini, chat interface) and inserted into the open-source
[Secworks AES core](https://github.com/secworks/aes). Each variant comes with its RTL, a testbench that fires the
trigger, the interaction log with the model, and Yosys / OpenSTA (SKY130) area and timing reports compared
against the clean design.

| Variant | Trigger | Payload | Area overhead | Timing |
|---|---|---|---|---|
| Trojan 1 | `block[15:0] == 0x6566` while `next` is asserted | output `key[255:128]` instead of the ciphertext (key leak) | +0.98% | slack met |
| Trojan 2 | 255th consecutive encryption (counter time-bomb) | replace `result[31:0]` with `key[31:0]` | +1.18% | slack met |
| Trojan 3 | three specific blocks in sequence (3-stage FSM, constants split into XOR/reduction logic) | lock `ready = 0` permanently (DoS) | **+0.41%** | slack met |

```
golden_metrics/              area + STA reports of the clean AES core
Trojan/TrojanN/aes_core.v    Trojan-inserted RTL (original Secworks license header kept)
Trojan/TrojanN/tb_aes_core.v testbench that exercises normal operation and fires the trigger
Trojan/TrojanN/metrics/      area + STA reports
Trojan/TrojanN/*.pdf         LLM interaction log
trojan_loop.py               Python model of the three triggers/payloads + PPA comparison (no simulator needed)
```

```bash
python trojan_loop.py                                                        # trigger models + PPA table
iverilog -o tb Trojan/Trojan3/tb_aes_core.v Trojan/Trojan3/aes_core.v <other Secworks AES sources> && vvp tb
```

The original submission write-up follows.

---

## Submission write-up: Creative AI-Driven Hardware Trojan Design and Optimization

## 1. AI Model & Interaction Framework
* **AI Model:** Gemini (Large Language Model by Google DeepMind)
* **Interaction Method:** Web-based Chat Interface (UI)
* **Interaction Mode:** Iterative Prompting, Contextual Feedback Loop, and Log-based Debugging

### Integrated Workflow:
* **Icarus Verilog:** Used for functional verification; AI assisted in interpreting simulation logs.
* **PPA Analysis Tools:** Used for Area, Timing, and Power reports; AI provided data-driven optimization strategies.

## 2. Methodology: How AI was Used to Modify the Code
This project followed a **"Heuristic Iterative Development Loop,"** where the AI acted as a "Security Architect" rather than a simple code generator.

### Phase 1: Conceptualization & Taxonomy Design
Instead of requesting code directly, the initial phase involved a collaborative brainstorming session with the AI.
* **AI Contribution:** Proposed multiple schemes ranging from counter-based triggers to complex "Shadow FSMs".
* **Creative Decision:** Selected a **Multi-Stage Sequence-Triggered DoS Trojan**. This is statistically impossible to occur during standard NIST test vectors.

### Phase 2: Logic Integration & Code Generation
I provided the original AES `aes_core.v` source code to the AI to implant the Trojan logic.
* **AI Contribution:** The AI identified the precise insertion points within the top-level control logic (`aes_core_ctrl`) and the register update block (`reg_update`). It generated a 3-stage FSM to track the sequence.

### Phase 3: Simulation-Driven Debugging
* **AI Contribution:** When the initial Testbench failed, the AI analyzed the raw simulation logs. It identified that the `ready` signal status was locked and provided a corrected Testbench sequence to validate the DoS state.

### Phase 4: PPA-Driven Optimization (The "Obfuscation" Phase)
* **AI Contribution:** The AI proposed a **Logic Obfuscation Strategy**.
* **Implementation:** Broke 128-bit constants into fragmented bitwise logical operations (XOR and reduction AND).
* **Result:** Reduced the total area overhead significantly between Trojan iterations.

## 3. PPA Impact Analysis: Golden vs. Trojans
Based on the synthesized metrics, the impact of Trojan insertion on the AES core was analyzed to ensure maximum stealth:

| Metric | Golden (Clean) | Trojan 1 | Trojan 2 | Trojan 3 (Optimized) |
| :--- | :--- | :--- | :--- | :--- |
| **Total Area** | 151,356.41 | 152,846.59 | 153,146.88 | **151,979.51** |
| **Area Overhead** | 0.00% | +0.98% | +1.18% | **+0.41%** |
| **Sequential Area**| 61,789.26 | 61,814.28 | 62,014.48 | 61,864.33 |
| **Timing Slack** | 80.2326 | 80.4417 | 80.5315 | 80.1413 |

**Key Observations:**
* **Stealth Optimization:** Trojan 3 achieved the lowest area overhead (+0.41%), making it virtually indistinguishable from standard synthesis tool variance.
* **Timing Stability:** All Trojan versions maintained a positive timing slack (MET), ensuring that the malicious logic did not compromise the operational frequency of the AES core.
* **Logic Efficiency:** By transitioning from raw 128-bit comparisons to obfuscated bitwise logic in Trojan 3, the combinational area was significantly optimized compared to Trojan 1 and 2.

## 4. Creative Highlights
* **Co-Design Partnership:** AI participated in a "Red Team vs. Blue Team" defense analysis to evade logic-based audits.
* **Data-Informed Refinement:** Used AI to parse complex PPA reports and achieve a balance between impact and physical stealth.

## 5. Conclusion
AI served as a **Chief Security Architect** and a **Back-end Optimization Engineer**, successfully elevating a functional requirement into an industrially optimized, stealthy hardware Trojan.