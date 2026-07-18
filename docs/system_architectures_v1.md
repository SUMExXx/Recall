# Recall — Architecture Diagrams

> Memory-layer detail (NMO schema, fixed-size chunking, 4-tier router,
> two-tier Matryoshka vectors, consolidation) lives in
> `plans/memory_engineering_v2.md` — that doc is the source of truth for
> everything inside the Memory Core boxes below.

## 1. System Topology

```mermaid
flowchart LR
    subgraph PHONE["OnePlus 15 — Snapdragon 8 Elite"]
        P1["Capture<br/>VAD gate, Opus stream"]
        P2["Ask / Memory UI<br/>recall notifications"]
        P3["Degraded mode<br/>Whisper-tiny + outbox"]
    end

    subgraph UNOQ["Arduino UNO Q"]
        A1["Linux: USB mic<br/>VAD + wake word"]
        A2["MCU: LEDs, bookmark,<br/>forget, hardware mute"]
    end

    subgraph PCC["Snapdragon X Elite PC — Hub"]
        C1["WS Hub + Device Registry<br/>mDNS, heartbeats, leases, resume tokens"]
        C2["NPU Workers<br/>Whisper-Base/Small-En · Nomic v1.5<br/>Qwen3-4B (GenieX) · Qwen3-Reranker"]
        C3["Memory Core (NMO)<br/>episodes + fixed chunks, vec int8[256],<br/>proactive recall, consolidation, policy engine"]
        C4["MCP Server"]
        C5["Dashboard<br/>canvas, transcript, health, bench"]
    end

    MCPC["Claude / any MCP client"]
    CLOUD["Qualcomm AI Cloud 100<br/>policy-gated batch only"]

    P1 -- "audio (Opus / WS)" --> C1
    A1 -- "wake + VAD events, audio" --> C1
    A2 -- "button events" --> C1
    C1 --> C2
    C2 --> C3
    C3 -- "proactive recall push" --> P2
    C1 -- "LED state topics" --> A2
    C3 --- C5
    MCPC <--> C4
    C4 --> C3
    C3 -. "encrypted batch jobs" .-> CLOUD

    P3 -. "PC down: local ASR + queue,<br/>sync on reconnect" .-> C1
    A1 -. "PC down: reroute audio" .-> P3
    C1 -. "heartbeats / leases" .- PHONE
    C1 -. "heartbeats / leases" .- UNOQ
```

## 2. Arduino UNO Q — Linux + MCU Split

```mermaid
flowchart TB
    PCHUB["PC Hub (primary)"]
    PHONE["Phone (fallback)"]

    subgraph LNX["Linux Side — Dragonwing QRB2210, Debian"]
        ALSA["ALSA Capture<br/>USB mic"]
        RING["Ring Buffer<br/>5 s pre-roll (wake word never clips speech)"]
        SVAD["Silero VAD<br/>ONNX"]
        WAKE["openWakeWord<br/>'Hey Recall'"]
        PUBW["Event Publisher<br/>WS client: vad / wake / button topics"]
        FWD["Audio Forwarder<br/>Opus over WS"]
        LBUF[("Standalone Buffer<br/>flush on reconnect")]
        SYSD["systemd Watchdog<br/>auto-restart services"]
    end

    subgraph MCUX["MCU Side — STM32U585, real-time"]
        LEDS["LED State Machine<br/>amber = capturing, blue = searching,<br/>green = recalled, red = muted / error"]
        BTNS["Buttons<br/>tap = bookmark, hold 3 s = forget last 5 min"]
        HMUTE["Hardware Mute Switch<br/>GPIO mic gate — physical privacy"]
        MWDT["MCU Watchdog<br/>error blink if Linux side goes silent"]
    end

    BRG["Bridge RPC<br/>internal serial link"]

    HMUTE --> ALSA
    ALSA --> RING
    RING --> SVAD
    SVAD --> WAKE
    WAKE --> PUBW
    SVAD --> PUBW
    RING --> FWD
    PUBW --> PCHUB
    FWD --> PCHUB
    PUBW -. "PC down: reroute" .-> PHONE
    FWD -. "PC down: reroute" .-> PHONE
    FWD -. "all peers down" .-> LBUF
    LBUF -. "on reconnect" .-> PCHUB
    BTNS --> BRG
    BRG --> PUBW
    PCHUB -- "state topics" --> PUBW
    PUBW --> BRG
    BRG --> LEDS
    SYSD -.-> PUBW & FWD
    MWDT -.-> LEDS
```

## 3. Phone App — Capture, Degraded Mode, Self-Heal

```mermaid
flowchart TB
    PCHUB["PC Hub<br/>WebSocket"]

    subgraph UIX["UI Layer"]
        CAPP["Capture Page<br/>auto-start on wake, manual toggle"]
        ASKP["Ask Page"]
        MEMP["Memory Page<br/>read / delete"]
        BAN["Mode Banner<br/>LIVE / DEGRADED"]
        NOTI["Proactive Recall<br/>notifications"]
    end

    subgraph CAPT["Capture Pipeline (on-device)"]
        MIC["Mic 16 kHz"]
        VAD["Silero VAD<br/>speech gating — only speech leaves the phone"]
        ENC["Opus Encoder"]
        STR["WS Streamer<br/>backpressure, seq numbers"]
    end

    subgraph DEG["Degraded-Mode Brain (PC down)"]
        TASR["Whisper-tiny INT8<br/>phone NPU (QNN / NNAPI)"]
        LDB[("Local Store<br/>SQLite, tombstoned deletes")]
        OBX2[("Outbox Queue")]
        LSR["Lite Search<br/>over synced cache"]
    end

    subgraph CONN["Connectivity + Self-Heal"]
        DISC["mDNS Discovery"]
        CMGR["Connection Manager<br/>heartbeat, exp backoff + jitter"]
        MODE["Mode Controller<br/>LIVE <-> DEGRADED auto-switch"]
        SYNC["Sync Manager<br/>content-hash dedupe, outbox replay"]
    end

    CAPP --> MIC
    MIC --> VAD
    VAD --> ENC
    ENC --> STR
    STR --> PCHUB
    VAD -. "PC unreachable" .-> TASR
    TASR --> LDB
    TASR --> OBX2
    SYNC --> OBX2
    SYNC -- "reconcile on reconnect" --> PCHUB
    DISC --> CMGR
    CMGR <--> PCHUB
    CMGR --> MODE
    MODE --> STR
    MODE -.-> TASR
    MODE --> BAN
    ASKP --> PCHUB
    ASKP -. "offline" .-> LSR
    LSR --> LDB
    MEMP --> PCHUB
    PCHUB -- "recall push" --> NOTI
```

## 4. PC Hub — Full Backend Architecture

```mermaid
flowchart TB
    PH["Phone (WebSocket)<br/>Opus audio / asks / sync"]
    AR["UNO Q (WebSocket)<br/>wake, VAD, button events / audio"]
    MCPC["MCP Clients<br/>Claude Desktop / Claude Code"]
    CLOUD["Qualcomm AI Cloud 100<br/>optional batch tier"]

    subgraph GW["Gateway Layer"]
        WSH["WebSocket Hub<br/>pub/sub topics, seq numbers, resume tokens"]
        REST["REST API<br/>memory CRUD, dump, health"]
        MCP["MCP Server (stdio / streamable HTTP)<br/>recall_search_memory, recall_ask,<br/>recall_bookmark_moment, recall_forget_range, recall_dump"]
    end

    subgraph NPU["Inference Workers — Hexagon NPU"]
        ASR["ASR: Whisper-Base-En w8a16 (live)<br/>+ Small-En (dream re-transcribe), ORT-QNN"]
        EMB["Embedder: Nomic-Embed-Text v1.5<br/>768-dim float → Matryoshka int8[256]"]
        LLM["LLM: Qwen3-4B-Instruct w4a16, GenieX<br/>+ Qwen3-Reranker-0.6B"]
    end

    subgraph CORE["Memory Core"]
        ING["Ingest + NMO Normalizer<br/>fixed 256-tok chunker, WAL journal"]
        RET["Retrieval Service<br/>4-tier router, brute-force vec,<br/>weighted RRF + rescore + rerank"]
        PRO["Proactive Recall Engine<br/>rolling 45s window, threshold, cooldown"]
        CON["Consolidation Agent<br/>dedupe, merge groups, compact, correct"]
        POL["Policy Engine<br/>privacy tags, cloud eligibility"]
    end

    subgraph ST["Local Storage"]
        VDB[("SQLite memory store<br/>NMO memories · episodes · chunks<br/>FTS5 + vec_chunks int8[256]")]
        GDB[("Graph + entity index<br/>bi-temporal relations, entity_mentions")]
        BLB[("Blob Store<br/>images, audio clips")]
        OBX[("Outbox / WAL")]
    end

    subgraph REL["Reliability Layer"]
        SUP["Supervisor<br/>health probes, auto-restart workers"]
        REG["Device Registry<br/>mDNS, heartbeats, leases"]
        MET["Metrics + Bench Logger<br/>RTF, latency, tok/s, NPU util"]
    end

    UI["Dashboard<br/>memory canvas, live transcript,<br/>ask panel, recall cards, device health"]

    PH --> WSH
    AR --> WSH
    WSH --> ING
    ING --> ASR
    ASR --> EMB
    EMB --> VDB & GDB
    ING --> OBX
    ING --> BLB
    ASR -- "rolling transcript" --> PRO
    PRO --> RET
    RET <--> VDB
    RET --> LLM
    PRO -- "recall event" --> WSH
    WSH -- "state topics, recall push" --> PH
    WSH -- "LED state" --> AR
    UI <--> WSH
    UI --> REST
    REST --> RET
    MCPC <--> MCP
    MCP --> RET
    MCP --> POL
    CON <--> VDB
    CON <--> GDB
    CON --> POL
    POL -. "encrypted batch jobs (optional)" .-> CLOUD
    SUP -.-> ASR & EMB & LLM
    REG <--> WSH
    MET --> UI
```
