# Laptop KWS; ESP32 LED only

Laptop captures its microphone, runs the frozen MP16 frontend and INT8 model,
then sends only nyala/mati/naik/turun text commands to ESP32. No audio, feature
extraction or model is run on the board. ESP32 controls PWM on GPIO25 and the
ready indicator on GPIO2; levels 1--5, default 3. All frozen MP artifacts remain
unchanged. A different microphone can change recognition quality.

Wiring: GPIO25 -> 330 ohm resistor -> LED anode; LED cathode -> GND.
No MAX9814/INMP441 wiring is needed. Board must remain connected to laptop USB.

```bash
cd /home/aliy/Coding/Myrepos/sound-processing-mini-project
~/.platformio/penv/bin/pio run -d firmware/esp32_led_only -t upload --upload-port /dev/ttyUSB0
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 .venv/bin/python tools/laptop_kws_led.py --port /dev/ttyUSB0
```

Default is prompted: ENTER first, then speak one command during the 1.2 second
capture. This mode has no ambient noise calibration. Ctrl+C stops and turns LED
off. Heartbeats keep the board alive while waiting at the prompt.

For hands-free commands, add --continuous. Stay quiet for about 2 seconds and
wait for [SIAP]. A fixed 1.2 second context includes up to 200 ms pre-roll;
after each command the gate waits for silence before rearming. The energy gate
is an operational policy, not model training. Laptop DSP/model are unchanged.
Use --seconds 30 with --continuous for a timed run. No speaker playback occurs.

Use --list-devices and --input-device ID for the internal microphone. PortAudio
must support mono 16 kHz. The LD_PRELOAD workaround is only for the known Conda
C++ library conflict on Linux x86-64. Without this conflict, omit the prefix.

Close other serial monitors first. Baud is 115200. Protocol lines: HELLO, PING,
READY 0/1, CMD sequence label. Replies are identity and ack JSON. Duplicate
sequence IDs do not apply brightness twice. Invalid commands do not act on LED.
Oversized lines are discarded. Missing heartbeats for 3 seconds turn LED off.

Logs go to results/device/laptop_model_led_*/: serial acknowledgements and host
predictions, with host latency clearly marked. No labelled accuracy/acceptance
is claimed; project_status and frozen metrics/manifests are not modified.
