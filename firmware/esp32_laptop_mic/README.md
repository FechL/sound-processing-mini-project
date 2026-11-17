# Laptop microphone -> ESP32 classification

Laptop captures mono PCM16 at 16 kHz and sends 800-sample (50 ms) blocks over USB UART
at 921600 baud using protocol 2 with CRC32 and per-packet acknowledgement.
Both firmware and host script must be updated together. ESP32 performs the frozen MP16 preprocessing, feature extraction,
INT8 classification, noise gate and LED actions. No model runs on the laptop.
No MAX9814/INMP441 is needed for this input mode.

External LED remains GPIO25 -> 330 ohm -> anode; cathode -> GND.
GPIO2 indicates ready. Brightness levels 1--5, default 3. nyala/mati control
ON/OFF; naik/turun change the stored level. Disconnecting the stream turns
the command LED off and resets the audio context and noise calibration.

From the repository root, close other serial programs and upload:

```bash
~/.platformio/penv/bin/pio run -d firmware/esp32_laptop_mic -t upload --upload-port /dev/ttyUSB0
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 .venv/bin/python tools/stream_laptop_to_esp32.py --port /dev/ttyUSB0
```

Stay quiet for about seven seconds, wait for [SIAP], then speak individual
nyala/mati/naik/turun commands with silence between them. Ctrl+C stops capture;
ESP32 fails safe after stream timeout. No speaker playback is enabled.
Use --list-devices and --input-device ID to select the internal microphone.
The device must support 16 kHz through PortAudio; no Python feature extraction,
resampling or inference is performed. Host queues are bounded; sequence gaps
reset the device context rather than classifying discontinuous audio.

Logs and host counters are stored in results/device/laptop_input_esp32_*/.
These are unlabelled operational logs, not MP16 acceptance or new accuracy.
The model was trained on MAX9814: laptop microphone recognition quality must
be checked separately. Original models, firmware and historical results remain
unchanged. The pre-build guard checks the frozen base package hashes.

Protocol 2: LPCM2!!! magic (8 bytes), little-endian uint32 sequence,
uint16 sample count, uint16 kind (0=PCM, 1=identity query), PCM16 payload,
then little-endian CRC32 of header+payload. Audio packets contain 800 samples,
1620 bytes total; identity queries contain zero samples, 20 bytes total.
UART input is framed binary only: a '?' byte in audio never triggers a command.
Output is JSON, including identity, ack, health, window and event packets.
The host waits for each ack and stops after a one-second ack timeout instead of
flooding a missing/unresponsive receiver. CRC failures, sequence gaps and input
overflows are shown separately in [USB] diagnostics. Higher PortAudio latency
and 50 ms callbacks reduce host scheduling pressure; no model gates are lowered.
Garbled boot-ROM text at a different baud is ignored, not mistaken for JSON.

To restore standalone MAX9814 input:

```bash
~/.platformio/penv/bin/pio run -d firmware/esp32 -t upload --upload-port /dev/ttyUSB0
```
